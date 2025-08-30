import json
import os
import re
import time

from dotenv import load_dotenv
from huggingface_hub import login
from keywords_examples import irrelevant_reviews, non_visitor_reviews, spam_reviews
from keywords_examples import irrelevant_keywords, non_visitor_keywords, spam_keywords
import pandas as pd
import torch
from transformers import AutoProcessor, AutoTokenizer, pipeline

load_dotenv()  # looks for .env in current dir
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.bfloat16  # bf16 on MPS can be flaky

model_id = "google/gemma-3-4b-it"

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
pipe = pipeline(
    task="image-text-to-text",
    model=model_id,
    processor=processor,          # forces fast processor, no warning
    torch_dtype=dtype,
    device=device                 # works for CPU/"mps"/cuda in recent Transformers
)

df = pd.read_csv("sample_input.csv")   # must have "review_text" column
df = df.head(10)

tokenizer = AutoTokenizer.from_pretrained(model_id)
POLICIES = ["ads", "irrelevant", "no_visit_rant", "no_violation"]

SYSTEM_PROMPT = (
    "You are a moderation system for business reviews.\n"
    "Classify the given review into one or more violation categories:\n"
    "- ads (advertisement or promotional content)\n"
    "- irrelevant (talks about unrelated topics)\n"
    "- no_visit_rant (complaints/rants without actual visit)\n"
    "- no_violation (valid review)\n\n"
    
    "Examples of ads violations:\n"
    f"{chr(10).join(['• ' + review[:100] + '...' for review in spam_reviews[:3]])}\n\n"
    
    "Examples of irrelevant violations:\n"
    f"{chr(10).join(['• ' + review[:100] + '...' for review in irrelevant_reviews[:3]])}\n\n"
    
    "Examples of no_visit_rant violations:\n"
    f"{chr(10).join(['• ' + review[:100] + '...' for review in non_visitor_reviews[:3]])}\n\n"
    
    "Respond with ONLY a JSON object in this exact format:\n"
    '{"violation": ["category1", "category2"], "rationale": "one sentence explanation of why you chose these categories"}\n'
    'If no violations, use: {"violation": [], "rationale": "one sentence explanation of why this is a valid review"}\n'
    'Keep rationale to one clear sentence.'
)

def build_chat_prompt(review_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Review:\n{review_text}"}]
        }
    ]
    # Convert to a single generation string using Gemma’s chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # adds assistant preamble so model continues correctly
    )

_JSON_PATTERN = re.compile(r"\{.*?\}", flags=re.DOTALL)

def _extract_first_json(s: str):
    m = _JSON_PATTERN.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

def _bool_map_from_list(labels):
    """Turn a list like ['ads', 'irrelevant'] into a full bool map with exclusivity rule for no_violation."""
    flags = {k: False for k in POLICIES}
    for lab in labels:
        if lab in flags and lab != "no_violation":
            flags[lab] = True
    # no_violation is True only if none of the violation flags are True
    flags["no_violation"] = not (flags["ads"] or flags["irrelevant"] or flags["no_visit_rant"])
    return flags

def rule_based_check(text: str):
    text_l = (text or "").lower()
    labels = []
    rationale_bits = []

    if any(re.search(p, text_l) for p in AD_PATTERNS):
        labels.append("ads")
        rationale_bits.append("Matched advertisement pattern")
    if any(re.search(p, text_l) for p in NO_VISIT_PATTERNS):
        labels.append("no_visit_rant")
        rationale_bits.append("Matched no-visit rant pattern")
    if any(re.search(p, text_l) for p in IRRELEVANT_PATTERNS):
        labels.append("irrelevant")
        rationale_bits.append("Matched irrelevant pattern")

    if not labels:
        return None

    flags = _bool_map_from_list(labels)
    return {
        "violations": flags,
        "confidence": 0.95 if labels else 0.0,
        "rationale": "; ".join(rationale_bits) or "Rule-based matched"
    }

def llm_classify(text: str):
    prompt = build_chat_prompt(text)
    out = pipe(
        prompt,
        max_new_tokens=128,
        return_full_text=False
    )
    raw = out[0]["generated_text"].strip()
    parsed = _extract_first_json(raw)

    if isinstance(parsed, dict) and isinstance(parsed.get("violation"), list):
        flags = _bool_map_from_list(parsed["violation"])
        rationale = parsed.get("rationale", "No rationale provided")
        return {
            "violations": flags,
            "confidence": 1.0,
            "rationale": rationale
        }
    else:
        # Fallback: treat as no_violation (nothing triggered)
        flags = _bool_map_from_list([])
        return {
            "violations": flags,
            "confidence": 0.0,
            "rationale": f"Unparseable model output: {raw[:120]}..."
        }

# ======================
# Full Pipeline
# ======================
pipeline_start_time = time.time()
outputs = []
for _, row in df.iterrows():
    review = row.get("text", "")
    # rb = rule_based_check(review)
    # result = rb if rb else llm_classify(review)
    result = llm_classify(review)
    outputs.append(result)

pipeline_end_time = time.time()
total_pipeline_time = pipeline_end_time - pipeline_start_time

# Expand four boolean columns
df["is_text_ads"] = [o["violations"]["ads"] for o in outputs]
df["is_text_irrelevant"] = [o["violations"]["irrelevant"] for o in outputs]
df["is_text_rant"] = [o["violations"]["no_visit_rant"] for o in outputs]

# Print timing statistics
print("=== TIMING STATISTICS ===")
print(f"Total pipeline time: {total_pipeline_time:.2f} seconds")
print(f"Number of reviews processed: {len(outputs)}")
print(f"Average time per review: {total_pipeline_time/len(outputs):.2f} seconds")

# Print rationales to screen
print("\n=== LLM Rationales ===")
for i, output in enumerate(outputs):
    print(f"Review {i+1}: {output['rationale']}")

df.to_csv("reviews_with_policy_flags.csv", index=False)
print(df[["text", "is_text_ads", "is_text_irrelevant", "is_text_rant"]].head(10))