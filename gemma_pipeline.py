# =============================================================================
# Business Review Moderation Pipeline using Google Gemma-3-4B-IT Model
# =============================================================================
# This pipeline analyzes business reviews for various policy violations and quality metrics
# using a multimodal LLM (Gemma-3-4B-IT) that can process both text and images.
#
# Key Features:
# - Text-based policy violation detection (ads, irrelevant content, no-visit rants)
# - Image analysis for advertisement and relevance detection
# - Review helpfulness evaluation
# - Sentiment-rating consistency analysis
# =============================================================================

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

# =============================================================================
# Environment Setup and Model Initialization
# =============================================================================

# Load environment variables from .env file for secure token management
# This prevents hardcoding sensitive information like API tokens
load_dotenv()  # looks for .env in current dir
hf_token = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face to access the Gemma model
# Required because Gemma is a gated model that requires acceptance of terms
login(token=hf_token)

# Device selection: Use MPS (Metal Performance Shaders) on Apple Silicon Macs for GPU acceleration
# Fall back to CPU if MPS is not available (e.g., on Intel Macs or other systems)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Use bfloat16 precision for memory efficiency and faster inference
# Note: bf16 on MPS can be flaky, but provides good balance of speed vs memory
dtype = torch.bfloat16

# Model selection: Google's Gemma-3-4B-IT (Instruction Tuned) model
# - 4B parameters: Good balance between performance and resource requirements
# - IT suffix: Instruction-tuned for better following prompts
# - Multimodal: Can process both text and images
model_id = "google/gemma-3-4b-it"

# Initialize the processor for handling multimodal inputs (text + images)
# use_fast=True ensures we use the faster tokenizer implementation
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# Create the pipeline for image-text-to-text generation
# This is the core component that will process our multimodal inputs
pipe = pipeline(
    task="image-text-to-text",  # Multimodal task: image + text → text
    model=model_id,
    processor=processor,          # forces fast processor, no warning
    torch_dtype=dtype,           # Use bfloat16 for memory efficiency
    device=device                 # works for CPU/"mps"/cuda in recent Transformers
)

# =============================================================================
# Data Loading and Configuration
# =============================================================================

# Load the validation dataset containing business reviews
# Must have "review_text" column for processing
df = pd.read_csv("sample_test.csv")

# Initialize tokenizer for converting chat templates to prompts
# This is needed for the chat-based prompting approach
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define the three main policy violation categories we're detecting
# These represent the core moderation rules for business reviews
POLICIES = ["ads", "irrelevant", "no_visit_rant"]

# =============================================================================
# Prompt Engineering for Different Analysis Tasks
# =============================================================================
# Each prompt is carefully crafted to:
# 1. Define the specific task clearly
# 2. Provide examples of violations
# 3. Request structured JSON output for easy parsing
# 4. Include business context for better decision making

# Prompt for analyzing images in reviews
# Determines if images are ads or relevant to the business
IMAGE_ANALYSIS_PROMPT = (
    "You are analyzing images from business reviews.\n"
    "For each image, determine:\n"
    "1. Is this image an advertisement? (ads, promotional content, marketing materials)\n"
    "2. Is this image relevant to the business/location? (directly related to the business/location, its services/function, or the customer/visitor experience)\n\n"
    "Respond with ONLY a JSON object: {\"is_ad\": true/false, \"is_relevant\": true/false}"
)

# Prompt for detecting advertisement content in review text
# Focuses on explicit marketing language and external business promotion
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

# Prompt for detecting irrelevant content
# Distinguishes between negative sentiment (relevant) and truly unrelated content
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

# Prompt for detecting reviews from non-visitors
# Important for maintaining review quality and authenticity
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

# Prompt for analyzing sentiment-rating consistency
# Helps identify potentially fake or misleading reviews
SENSIBILITY_PROMPT = (
    "You are a moderation system for business reviews.\n"
    "Determine if the given review sentiment aligns with the provided rating.\n"
    "The rating is from 1 to 5, where 1 is the worst and 5 is the best.\n"
    "Respond with ONLY a JSON object: {\"is_sensible\": true/false}"
)

# =============================================================================
# Utility Functions
# =============================================================================

def download_image(url):
    """
    Download image from URL and return PIL Image object
    
    Args:
        url (str): URL of the image to download
        
    Returns:
        PIL.Image.Image or None: Downloaded image or None if download fails
        
    Note: This function handles network errors gracefully and sets a reasonable timeout
    to prevent the pipeline from hanging on slow or broken image URLs.
    """
    try:
        response = requests.get(url, timeout=10)  # 10 second timeout prevents hanging
        response.raise_for_status()  # Raise exception for HTTP errors
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

def build_image_analysis_prompt(review_text: str, business_info: str, image) -> str:
    """
    Build a multimodal prompt for image analysis using Gemma's chat template format
    
    Args:
        review_text (str): The review text to provide context
        business_info (str): Business location information for context
        image: PIL Image object to analyze
        
    Returns:
        str: Formatted prompt string ready for the pipeline
        
    Note: This function uses Gemma's chat template to format the prompt in a way
    that the model expects, ensuring optimal performance and response quality.
    """
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
    # This ensures the model receives the prompt in its expected format
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# Regular expression pattern for extracting JSON responses from model output
# Uses DOTALL flag to match across multiple lines
_JSON_PATTERN = re.compile(r"\{.*?\}", flags=re.DOTALL)

def _extract_first_json(s: str):
    """
    Extract the first JSON object from a string using regex
    
    Args:
        s (str): String that may contain JSON
        
    Returns:
        dict: Parsed JSON object or empty dict if parsing fails
        
    Note: This function is robust to cases where the model outputs additional text
    before or after the JSON response, which is common with LLMs.
    """
    m = _JSON_PATTERN.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

def _bool_map_from_list(labels):
    """
    Turn a list like ['ads', 'irrelevant'] into a full bool map with exclusivity rule for no_violation.
    
    Args:
        labels (list): List of violation labels
        
    Returns:
        dict: Boolean map with all policy flags and no_violation flag
        
    Note: The no_violation flag is True only if none of the violation flags are True,
    implementing an exclusivity rule that ensures each review is classified exactly once.
    """
    flags = {k: False for k in POLICIES}
    for lab in labels:
        if lab in flags and lab != "no_violation":
            flags[lab] = True
    # no_violation is True only if none of the violation flags are True
    flags["no_violation"] = not (flags["ads"] or flags["irrelevant"] or flags["no_visit_rant"])
    return flags

def create_business_info(row):
    """
    Create business information string from row data for context
    
    Args:
        row: Pandas row containing business metadata
        
    Returns:
        str: Formatted business information string
        
    Note: This function provides essential context to the LLM for making informed
    decisions about review relevance and violations. Business context helps distinguish
    between legitimate complaints and truly irrelevant content.
    """
    parts = []
    
    if pd.notna(row.get("name")) and str(row.get("name")).strip():
        parts.append(f"Location Name: {str(row.get('name')).strip()}")
    
    if pd.notna(row.get("category")) and str(row.get("category")).strip():
        parts.append(f"Category: {str(row.get('category')).strip()}")
    
    if not parts:
        return "Location Information: Not available"
    
    return "\n".join(parts)

# =============================================================================
# LLM Analysis Functions
# =============================================================================
# Each function follows the same pattern:
# 1. Build a prompt with business context and review text
# 2. Use the chat template for proper formatting
# 3. Call the pipeline with controlled token generation
# 4. Parse JSON response with fallback handling
# 5. Return structured results with confidence scores

def llm_evaluate_helpfulness(review_text: str, business_info: str) -> str:
    """
    Use LLM to evaluate overall helpfulness based on added value compared to basic Google info
    
    Args:
        review_text (str): The review text to evaluate
        business_info (str): Business context information
        
    Returns:
        str: Helpfulness rating (not_helpful, helpful, or very_helpful)
        
    Note: This function evaluates whether a review provides value beyond basic business
    information that users could find elsewhere. This is crucial for maintaining review
    quality and helping users make informed decisions.
    """
    
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
    
    # Generate response with controlled token limit for efficiency
    out = pipe(
        prompt,
        max_new_tokens=128,  # Limit output length for faster processing
        return_full_text=False
    )
    raw = out[0]["generated_text"].strip()
    parsed = _extract_first_json(raw)
    
    if isinstance(parsed, dict) and parsed.get("helpfulness"):
        return parsed["helpfulness"]
    else:
        # Fallback to default value if parsing fails
        # This ensures the pipeline continues even with malformed responses
        return "not_helpful"

def llm_classify_ads(text: str, business_info: str):
    """
    Classify if review contains advertisement content
    
    Args:
        text (str): Review text to analyze
        business_info (str): Business context information
        
    Returns:
        dict: Classification result with confidence score
        
    Note: Advertisement detection is critical for maintaining review authenticity.
    Reviews should reflect genuine customer experiences, not marketing campaigns.
    """
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
        # Fallback: assume no violation if parsing fails
        # This is a conservative approach that prioritizes false negatives over false positives
        return {
            "is_ad": False,
            "confidence": 0.0
        }

def llm_classify_irrelevant(text: str, business_info: str):
    """
    Classify if review text is irrelevant to the business
    
    Args:
        text (str): Review text to analyze
        business_info (str): Business context information
        
    Returns:
        dict: Classification result with confidence score
        
    Note: Distinguishing between negative sentiment and irrelevant content is crucial.
    A negative review about service quality is relevant; a review about politics is not.
    """
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
    """
    Classify if review is from someone who never visited
    
    Args:
        text (str): Review text to analyze
        business_info (str): Business context information
        
    Returns:
        dict: Classification result with confidence score
        
    Note: This detection is essential for review authenticity. Reviews should be
    based on actual experiences, not hearsay or external opinions.
    """
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
    """
    Classify if review sentiment aligns with rating
    
    Args:
        text (str): Review text to analyze
        business_info (str): Business context information
        rating (float): Numerical rating (1-5) given by the user
        
    Returns:
        dict: Classification result with confidence score
        
    Note: This helps identify potentially fake reviews where sentiment and rating
    are inconsistent, which could indicate review manipulation or fake accounts.
    """
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
        # Default to sensible if parsing fails
        # This prevents false positives in sensibility detection
        return {
            "is_sensible": True,
            "confidence": 0.0
        }

def llm_analyze_image(review_text: str, business_info: str, image_url: str):
    """
    Analyze a single image for ads and relevance
    
    Args:
        review_text (str): Review text for context
        business_info (str): Business information for context
        image_url (str): URL of the image to analyze
        
    Returns:
        dict: Analysis results with boolean flags
        
    Note: Image analysis is computationally expensive, so we limit processing
    to the first few images per review and handle download failures gracefully.
    """
    if not image_url or pd.isna(image_url):
        return {
            "is_ad": False,
            "is_relevant": False
        }
    
    # Download and process the image
    image = download_image(image_url)
    if image is None:
        return {
            "is_ad": False,
            "is_relevant": False
        }
    
    # Build multimodal prompt and analyze
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
    """
    Process all images for a single review
    
    Args:
        review_text (str): Review text for context
        business_info (str): Business information for context
        pics_collapsed: String representation of image URLs or list of URLs
        
    Returns:
        dict: Aggregated image analysis results
        
    Note: This function processes multiple images per review and aggregates the results.
    We limit processing to the first 3 images to balance accuracy with performance.
    """
    if pd.isna(pics_collapsed) or not pics_collapsed:
        return {
            "is_image_ad": False,
            "is_image_irrelevant": False,
            "image_analysis": []
        }
    
    try:
        # Parse the pics_collapsed string (assuming it's a string representation of a list)
        # Handle different input formats gracefully
        if isinstance(pics_collapsed, str):
            # Remove brackets and quotes, split by comma
            urls = [url.strip().strip("'\"") for url in pics_collapsed.strip("[]").split(",")]
        else:
            urls = pics_collapsed
        
        image_results = []
        any_ad = False
        any_relevant = False
        
        # Process only the first 3 images to limit processing time
        # This is a trade-off between accuracy and performance
        for i, url in enumerate(urls[:3]):
            if url and url.strip():
                result = llm_analyze_image(review_text, business_info, url.strip())
                image_results.append(result)
                
                # Track if any image violates policies
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
        # Return safe defaults on error to prevent pipeline failure
        return {
            "is_image_ad": False,
            "is_image_irrelevant": False,
            "image_analysis": []
        }

# =============================================================================
# Main Pipeline Execution
# =============================================================================
# The pipeline processes each review through multiple classification stages:
# 1. Text-based policy violation detection
# 2. Image analysis for visual content
# 3. Helpfulness evaluation
# 4. Results aggregation and output generation

# Record start time for performance monitoring
pipeline_start_time = time.time()

# Initialize output containers for each classification type
# This allows us to process all reviews first, then add results to the dataframe
ads_outputs = []
irrelevant_outputs = []
no_visit_rant_outputs = []
sensibility_outputs = []
image_outputs = []

# Process each review in the dataset
# This is the core loop where all LLM analysis happens
for _, row in df.iterrows():
    # Extract review components
    review = row.get("text", "")
    pics = row.get("pics_collapsed", "")
    rating = row.get("rating", 3.0)  # Default to 3.0 if no rating
    business_info = create_business_info(row)
    
    # Process each classification separately for modularity and error isolation
    # If one classification fails, others can still succeed
    ads_result = llm_classify_ads(review, business_info)
    irrelevant_result = llm_classify_irrelevant(review, business_info)
    no_visit_rant_result = llm_classify_no_visit_rant(review, business_info)
    sensibility_result = llm_classify_sensibility(review, business_info, rating)
    
    # Store results for later processing
    ads_outputs.append(ads_result)
    irrelevant_outputs.append(irrelevant_result)
    no_visit_rant_outputs.append(no_visit_rant_result)
    sensibility_outputs.append(sensibility_result)
    
    # Process image analysis (most computationally expensive step)
    image_result = process_images_for_review(review, business_info, pics)
    image_outputs.append(image_result)

# Record end time and calculate total processing time
pipeline_end_time = time.time()
total_pipeline_time = pipeline_end_time - pipeline_start_time

# =============================================================================
# Results Processing and DataFrame Updates
# =============================================================================
# Convert LLM outputs to boolean columns for easy analysis and filtering

# Create boolean columns for text-based violations
df["is_text_ad"] = [o["is_ad"] for o in ads_outputs]
df["is_text_rant"] = [o["is_no_visit_rant"] for o in no_visit_rant_outputs]

# Create boolean columns for image analysis results
df["is_image_ad"] = [o["is_image_ad"] for o in image_outputs]
df["is_image_irrelevant"] = [o["is_image_irrelevant"] for o in image_outputs]

# Create text irrelevance column
df["is_text_irrelevant"] = [o["is_irrelevant"] for o in irrelevant_outputs]

# Create sensibility column for sentiment-rating consistency
df["sensibility"] = [o["is_sensible"] for o in sensibility_outputs]

# Create helpfulness column based on LLM evaluation of added value
# This is a separate pass through the data for the helpfulness metric
df["helpfulness"] = [llm_evaluate_helpfulness(row.get("text", ""), create_business_info(row)) for _, row in df.iterrows()]

# =============================================================================
# Performance Reporting and Output
# =============================================================================

# Print timing statistics for performance monitoring
# This helps identify bottlenecks and optimize the pipeline
print("=== TIMING STATISTICS ===")
print(f"Total pipeline time: {total_pipeline_time:.2f} seconds")
print(f"Number of reviews processed: {len(ads_outputs)}")
print(f"Average time per review: {total_pipeline_time/len(ads_outputs):.2f} seconds")

# Save the processed results to CSV for further analysis
# This preserves all original data plus the new classification columns
df.to_csv("pipeline_output.csv", index=False)

# Display a preview of the results for verification
print("\n=== Final Output Preview ===")
print(df[["text", "is_text_ad", "is_text_rant", "is_text_irrelevant", "sensibility", "is_image_ad", "is_image_irrelevant", "helpfulness"]].head(10))

# Provide detailed column descriptions for users
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