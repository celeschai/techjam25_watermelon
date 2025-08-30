import json
import os
import re
import time
import requests
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
from huggingface_hub import login
from examples import irrelevant_reviews, non_visitor_reviews, spam_reviews
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

# Read from dashboard input
try:
    df = pd.read_csv("dashboard.csv")
    print(f"Successfully loaded dashboard.csv with {len(df)} rows")
except FileNotFoundError:
    print("Error: dashboard.csv not found. Please ensure the dashboard has uploaded a CSV file.")
    exit(1)
except Exception as e:
    print(f"Error reading dashboard.csv: {e}")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_id)
POLICIES = ["ads", "irrelevant", "no_visit_rant"]



IMAGE_ANALYSIS_PROMPT = (
    "You are analyzing images from business reviews.\n"
    "For each image, determine:\n"
    "1. Is this image an advertisement? (ads, promotional content, marketing materials)\n"
    "2. Is this image relevant to the business/location? (directly related to the business/location, its services/function, or the customer/visitor experience)\n\n"
    "Respond with ONLY a JSON object: {\"is_ad\": true/false, \"is_relevant\": true/false}"
)

ADS_PROMPT = (
    "You are a moderation system for business reviews.\n"
    "Determine if the given review contains explicit advertisement content.\n\n"
    "Clear signs of ads violations:\n"
    "- Provides website, phone number, or contact information\n"
    "- Promotes external businesses or services\n"
    "- Contains marketing language or promotional content\n"
    "- Offers discounts, deals, or promotional codes\n\n"
    "Examples of ads violations:\n"
    f"{chr(10).join(['• ' + review[:100] + '...' for review in spam_reviews[:3]])}\n\n"
    "Respond with ONLY a JSON object: {\"is_ad\": true/false}"
)

IRRELEVANT_PROMPT = (
    "You are a moderation system for business reviews.\n"
    "Determine if the given review text is irrelevant to the business.\n\n"
    "User sentiment is considered relevant.\n"
    "A review is irrelevant if it:\n"
    "- Only talks about unrelated topics\n"
    "- Discusses personal issues not related to the business\n"
    "- Comments on unrelated events or situations\n"
    "Examples of irrelevant violations:\n"
    f"{chr(10).join(['• ' + review[:100] + '...' for review in irrelevant_reviews[:3]])}\n\n"
    "Respond with ONLY a JSON object: {\"is_irrelevant\": true/false}"
)

NO_VISIT_RANT_PROMPT = (
    "You are a moderation system for business reviews.\n"
    "Determine if the given review is from someone who never visited the place.\n\n"
    "User expressing their sentiment does not mean they have never visited"
    "A no_visit_rant violation occurs when the review:\n"
    "- Explicitly states the person has never been to the place\n"
    "- Contains second-hand information or hearsay\n"
    "- Is based on external reports or rumors\n"
    "- Expresses opinions without firsthand experience\n\n"
    "Examples of no_visit_rant violations:\n"
    f"{chr(10).join(['• ' + review[:100] + '...' for review in non_visitor_reviews[:10]])}\n\n"
    "Keywords: never been here, I haven't visited, looks awful from the outside, from what I hear, I'm not going to visit, I've decided not to go, from the looks of it, I heard a rumor, I just drove by\n"
    "Respond with ONLY a JSON object: {\"is_no_visit_rant\": true/false}"
)

SENSIBILITY_PROMPT = (
    "You are a moderation system for business reviews.\n"
    "Determine if the given review sentiment aligns with the provided rating.\n"
    "The rating is from 1 to 5, where 1 is the worst and 5 is the best.\n"
    "Respond with ONLY a JSON object: {\"is_sensible\": true/false}"
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
        "- Specific descriptions of services/products, such as quality of food/items\n"
        "- Extra information not found in basic business listings\n"
        "- Personal insights that help with decision-making\n\n"
        "Rate the helpfulness:\n"
        "- not_helpful: review mentions the business but provides no useful information beyond basic facts, reading this review is a waste of time\n"
        "- helpful: provides some helpful information about the business that would help with visit decisions\n"
        "- very_helpful: gives detailed, specific information about the food/item/service quality or experience that significantly helps with visit decisions\n\n"
        "Respond with ONLY a JSON object: {\"helpfulness\": \"rating\"}"
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

def llm_classify_ads(text: str, business_info: str):
    """Classify if review contains advertisement content"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ADS_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Location Information:\n{business_info}\n\nReview:\n{text}"}]
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

    if isinstance(parsed, dict) and isinstance(parsed.get("is_ad"), bool):
        return {
            "is_ad": parsed["is_ad"],
            "confidence": 1.0
        }
    else:
        return {
            "is_ad": False,
            "confidence": 0.0
        }

def llm_classify_irrelevant(text: str, business_info: str):
    """Classify if review text is irrelevant to the business"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": IRRELEVANT_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Location Information:\n{business_info}\n\nReview:\n{text}"}]
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

    if isinstance(parsed, dict) and isinstance(parsed.get("is_irrelevant"), bool):
        return {
            "is_irrelevant": parsed["is_irrelevant"],
            "confidence": 1.0
        }
    else:
        return {
            "is_irrelevant": False,
            "confidence": 0.0
        }

def llm_classify_no_visit_rant(text: str, business_info: str):
    """Classify if review is from someone who never visited"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": NO_VISIT_RANT_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Location Information:\n{business_info}\n\nReview:\n{text}"}]
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

    if isinstance(parsed, dict) and isinstance(parsed.get("is_no_visit_rant"), bool):
        return {
            "is_no_visit_rant": parsed["is_no_visit_rant"],
            "confidence": 1.0
        }
    else:
        return {
            "is_no_visit_rant": False,
            "confidence": 0.0
        }

def llm_classify_sensibility(text: str, business_info: str, rating: float):
    """Classify if review sentiment aligns with rating"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SENSIBILITY_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Location Information:\n{business_info}\n\nReview:\n{text}\n\nRating: {rating}/5"}]
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

    if isinstance(parsed, dict) and isinstance(parsed.get("is_sensible"), bool):
        return {
            "is_sensible": parsed["is_sensible"],
            "confidence": 1.0
        }
    else:
        return {
            "is_sensible": True,
            "confidence": 0.0
        }

def llm_analyze_image(review_text: str, business_info: str, image_url: str):
    """Analyze a single image for ads and relevance"""
    if not image_url or pd.isna(image_url):
        return {
            "is_ad": False,
            "is_relevant": False
        }
    
    image = download_image(image_url)
    if image is None:
        return {
            "is_ad": False,
            "is_relevant": False
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
            "is_relevant": parsed.get("is_relevant", False)
        }
    else:
        return {
            "is_ad": False,
            "is_relevant": False
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
        any_relevant = False
        
        # Process only the first 3 images to limit processing time
        for i, url in enumerate(urls[:3]):
            if url and url.strip():
                result = llm_analyze_image(review_text, business_info, url.strip())
                image_results.append(result)
                
                if result["is_ad"]:
                    any_ad = True
                if result["is_relevant"]:
                    any_relevant = True
        
        return {
            "is_image_ad": any_ad,
            "is_image_irrelevant": len(image_results) > 0 and not any_relevant,
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
ads_outputs = []
irrelevant_outputs = []
no_visit_rant_outputs = []
sensibility_outputs = []
image_outputs = []



for _, row in df.iterrows():
    review = row.get("text", "")
    pics = row.get("pics_collapsed", "")
    rating = row.get("rating", 3.0)  # Default to 3.0 if no rating
    business_info = create_business_info(row)
    
    # Process each classification separately
    ads_result = llm_classify_ads(review, business_info)
    irrelevant_result = llm_classify_irrelevant(review, business_info)
    no_visit_rant_result = llm_classify_no_visit_rant(review, business_info)
    sensibility_result = llm_classify_sensibility(review, business_info, rating)
    
    ads_outputs.append(ads_result)
    irrelevant_outputs.append(irrelevant_result)
    no_visit_rant_outputs.append(no_visit_rant_result)
    sensibility_outputs.append(sensibility_result)
    
    # Process image analysis
    image_result = process_images_for_review(review, business_info, pics)
    image_outputs.append(image_result)

pipeline_end_time = time.time()
total_pipeline_time = pipeline_end_time - pipeline_start_time

# Create boolean columns for violations
df["is_text_ad"] = [o["is_ad"] for o in ads_outputs]
df["is_text_rant"] = [o["is_no_visit_rant"] for o in no_visit_rant_outputs]

# Create boolean columns for image analysis
df["is_image_ad"] = [o["is_image_ad"] for o in image_outputs]
df["is_image_irrelevant"] = [o["is_image_irrelevant"] for o in image_outputs]

# Create text irrelevance column
df["is_text_irrelevant"] = [o["is_irrelevant"] for o in irrelevant_outputs]

# Create sensibility column
df["sensibility"] = [o["is_sensible"] for o in sensibility_outputs]

# Create helpfulness column based on LLM evaluation of added value
df["helpfulness"] = [llm_evaluate_helpfulness(row.get("text", ""), create_business_info(row)) for _, row in df.iterrows()]

# Print timing statistics
print("=== TIMING STATISTICS ===")
print(f"Total pipeline time: {total_pipeline_time:.2f} seconds")
print(f"Number of reviews processed: {len(ads_outputs)}")
print(f"Average time per review: {total_pipeline_time/len(ads_outputs):.2f} seconds")



# Save the pipeline output
df.to_csv("dashboard_output.csv", index=False)

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