import json
import os
import re
import time
import requests
from PIL import Image
from io import BytesIO

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
    '{"violation": ["category1", "category2"], "rationale": "one sentence explanation of why you chose these categories", "is_text_irrelevant": true/false, "sensibility": true/false}\n'
    'If no violations, use: {"violation": [], "rationale": "one sentence explanation of why this is a valid review", "is_text_irrelevant": true/false, "sensibility": true/false}\n'
    'Keep rationale to one clear sentence. For is_text_irrelevant, evaluate if the text is irrelevant or unrelated to the business/location info:\n'
    '- true: text is irrelevant or unrelated to the business/location\n'
    '- false: text is relevant and related to the business/location\n\n'
    'For sensibility: Given the rating out of 5, does the user\'s attitude through the text actually align with their rating? true if aligned, false if not aligned.'
)

IMAGE_ANALYSIS_PROMPT = (
    "You are analyzing images from business reviews.\n"
    "For each image, determine:\n"
    "1. Is this image an advertisement? (ads, promotional content, marketing materials)\n"
    "2. Is this image relevant to the business/location? (directly related to the business/location, its services/function, or the customer/visitor experience)\n\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    '{"is_ad": true/false, "is_relevant": true/false, "rationale": "one sentence explanation"}\n'
    'Keep rationale to one clear sentence explaining your classification. For relevance, check how well the image relates to the actual business/location info provided(if none is provided, say true).'
)

def download_image(url):
    """Download image from URL and return PIL Image object"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

def build_chat_prompt(review_text: str, business_info: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Location Information:\n{business_info}\n\nReview:\n{review_text}"}]
        }
    ]
    # Convert to a single generation string using Gemma's chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # adds assistant preamble so model continues correctly
    )

def build_image_analysis_prompt(review_text: str, business_info: str, image) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": IMAGE_ANALYSIS_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Location Information:\n{business_info}\n\nReview text:\n{review_text}\n\nAnalyze the attached image:"},
                {"type": "image", "image": image}
            ]
        }
    ]
    # Convert to a single generation string using Gemma's chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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

def create_business_info(row):
    """Create business information string from row data"""
    parts = []
    
    if pd.notna(row.get("name")) and str(row.get("name")).strip():
        parts.append(f"Location Name: {str(row.get('name')).strip()}")
    
    if pd.notna(row.get("category")) and str(row.get("category")).strip():
        parts.append(f"Category: {str(row.get('category')).strip()}")
    
    if pd.notna(row.get("description")) and str(row.get("description")).strip():
        parts.append(f"Description: {str(row.get('description')).strip()}")
    
    if not parts:
        return "Location Information: Not available"
    
    return "\n".join(parts)

def llm_evaluate_helpfulness(review_text: str, business_info: str) -> str:
    """Use LLM to evaluate overall helpfulness based on added value compared to basic Google info"""
    
    HELPFULNESS_PROMPT = (
        "You are evaluating how useful a business review is to someone considering visiting this place.\n"
        "Consider: Does this review add value compared to what a basic Google introduction would contain?\n\n"
        "A helpful review should contain:\n"
        "- Visiting experience details\n"
        "- Specific descriptions of services/products\n"
        "- Extra information not found in basic business listings\n"
        "- Personal insights that help with decision-making\n\n"
        "Rate the helpfulness:\n"
        "- not_helpful: review mentions the business but provides no useful information beyond basic facts, reading this review is a waste of time\n"
        "- helpful: provides some helpful information about the business that would help with visit decisions\n"
        "- very_helpful: gives detailed, specific information about the business that significantly helps with visit decisions\n\n"
        "Respond with ONLY a JSON object: {\"helpfulness\": \"rating\", \"rationale\": \"explanation\"}"
    )
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": HELPFULNESS_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Location Information:\n{business_info}\n\nReview Text:\n{review_text}\n\nEvaluate the overall helpfulness of this review."}]
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    out = pipe(
        prompt,
        max_new_tokens=128,
        return_full_text=False
    )
    raw = out[0]["generated_text"].strip()
    parsed = _extract_first_json(raw)
    
    if isinstance(parsed, dict) and parsed.get("helpfulness"):
        return parsed["helpfulness"]
    else:
        # Fallback to default value if parsing fails
        return "not_helpful"

def llm_classify_text(text: str, business_info: str):
    prompt = build_chat_prompt(text, business_info)
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
        is_text_irrelevant = parsed.get("is_text_irrelevant", False)
        sensibility = parsed.get("sensibility", False)
        return {
            "violations": flags,
            "confidence": 1.0,
            "rationale": rationale,
            "is_text_irrelevant": is_text_irrelevant,
            "sensibility": sensibility
        }
    else:
        # Fallback: treat as no_violation (nothing triggered)
        flags = _bool_map_from_list([])
        return {
            "violations": flags,
            "confidence": 0.0,
            "rationale": f"Unparseable model output: {raw[:120]}...",
            "is_text_irrelevant": False,
            "sensibility": False
        }

def llm_analyze_image(review_text: str, business_info: str, image_url: str):
    """Analyze a single image for ads and relevance"""
    if not image_url or pd.isna(image_url):
        return {
            "is_ad": False,
            "is_relevant": False,
            "rationale": "No image provided"
        }
    
    image = download_image(image_url)
    if image is None:
        return {
            "is_ad": False,
            "is_relevant": False,
            "rationale": "Failed to download image"
        }
    
    prompt = build_image_analysis_prompt(review_text, business_info, image)
    out = pipe(
        prompt,
        max_new_tokens=128,
        return_full_text=False
    )
    raw = out[0]["generated_text"].strip()
    parsed = _extract_first_json(raw)
    
    if isinstance(parsed, dict):
        return {
            "is_ad": parsed.get("is_ad", False),
            "is_relevant": parsed.get("is_relevant", False),
            "rationale": parsed.get("rationale", "No rationale provided")
        }
    else:
        return {
            "is_ad": False,
            "is_relevant": False,
            "rationale": f"Unparseable model output: {raw[:120]}..."
        }

def process_images_for_review(review_text: str, business_info: str, pics_collapsed):
    """Process all images for a single review"""
    if pd.isna(pics_collapsed) or not pics_collapsed:
        return {
            "is_image_ad": False,
            "is_image_irrelevant": False,
            "image_analysis": []
        }
    
    try:
        # Parse the pics_collapsed string (assuming it's a string representation of a list)
        if isinstance(pics_collapsed, str):
            # Remove brackets and quotes, split by comma
            urls = [url.strip().strip("'\"") for url in pics_collapsed.strip("[]").split(",")]
        else:
            urls = pics_collapsed
        
        image_results = []
        any_ad = False
        any_irrelevant = False
        
        for url in urls:
            if url and url.strip():
                result = llm_analyze_image(review_text, business_info, url.strip())
                image_results.append(result)
                
                if result["is_ad"]:
                    any_ad = True
                if not result["is_relevant"]:
                    any_irrelevant = True
        
        return {
            "is_image_ad": any_ad,
            "is_image_irrelevant": any_irrelevant,
            "image_analysis": image_results
        }
    
    except Exception as e:
        print(f"Error processing images: {e}")
        return {
            "is_image_ad": False,
            "is_image_irrelevant": False,
            "image_analysis": []
        }

# ======================
# Full Pipeline
# ======================
pipeline_start_time = time.time()
outputs = []
image_outputs = []

for _, row in df.iterrows():
    review = row.get("text", "")
    pics = row.get("pics_collapsed", "")
    business_info = create_business_info(row)
    
    # Process text classification
    text_result = llm_classify_text(review, business_info)
    outputs.append(text_result)
    
    # Process image analysis
    image_result = process_images_for_review(review, business_info, pics)
    image_outputs.append(image_result)

pipeline_end_time = time.time()
total_pipeline_time = pipeline_end_time - pipeline_start_time

# Create boolean columns for violations
df["is_text_ad"] = [o["violations"]["ads"] for o in outputs]
df["is_text_rant"] = [o["violations"]["no_visit_rant"] for o in outputs]

# Create boolean columns for image analysis
df["is_image_ad"] = [o["is_image_ad"] for o in image_outputs]
df["is_image_irrelevant"] = [o["is_image_irrelevant"] for o in image_outputs]

# Create text irrelevance column
df["is_text_irrelevant"] = [o["is_text_irrelevant"] for o in outputs]

# Create sensibility column
df["sensibility"] = [o["sensibility"] for o in outputs]

# Create helpfulness column based on LLM evaluation of added value
df["helpfulness"] = [llm_evaluate_helpfulness(row.get("text", ""), create_business_info(row)) for _, row in df.iterrows()]

# Print timing statistics
print("=== TIMING STATISTICS ===")
print(f"Total pipeline time: {total_pipeline_time:.2f} seconds")
print(f"Number of reviews processed: {len(outputs)}")
print(f"Average time per review: {total_pipeline_time/len(outputs):.2f} seconds")

# Print rationales to screen
print("\n=== LLM Rationales ===")
for i, (text_output, image_output) in enumerate(zip(outputs, image_outputs)):
    print(f"Review {i+1} - Text: {text_output['rationale']}")
    for j, img_result in enumerate(image_output['image_analysis']):
        print(f"  Image {j+1}: {img_result['rationale']}")

df.to_csv("reviews_with_policy_flags.csv", index=False)

print("\n=== Final Output Preview ===")
print(df[["text", "is_text_ad", "is_text_rant", "is_text_irrelevant", "sensibility", "is_image_ad", "is_image_irrelevant", "helpfulness"]].head(10))
print("\n=== Column Descriptions ===")
print("- is_text_ad: Boolean - review contains advertisement content")
print("- is_text_rant: Boolean - review is a rant without actual visit")
print("- is_text_irrelevant: Boolean - is the text irrelevant or unrelated to the business/location info?")
print("- sensibility: Boolean - does the user's attitude in the text align with their rating?")
print("- is_image_ad: Boolean - images contain advertisement content")
print("- is_image_irrelevant: Boolean - images are irrelevant to the business/location info")
print("- helpfulness: Overall helpfulness rating - does the review add value compared to basic Google info?")
print("  • not_helpful: review mentions the business but provides no useful information beyond basic facts, reading this review is a waste of time")
print("  • helpful: provides some helpful information about the business that would help with visit decisions")
print("  • very_helpful: gives detailed, specific information about the business that significantly helps with visit decisions")