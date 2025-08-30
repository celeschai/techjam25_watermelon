import ast
import pandas as pd
test_file = pd.read_csv('vt_merged.csv').drop('Unnamed: 0', axis = 1).iloc[17]
print(test_file)

from huggingface_hub import login
import os
from dotenv import load_dotenv

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

from transformers import pipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "google/gemma-3-4b-it"

pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    device=device,          
    torch_dtype=torch.bfloat16
)

# Testing ads + description

advertisement_examples = [
    "Billboard with product name and price",
    "Social media post promoting a sale",
    "Banner showing a company logo with a slogan",
    "Flyer with a discount coupon"
]

for idx, row in test_file.iterrows():
    test_list = ast.literal_eval(row['pics_collapsed'])
    if test_list:         
        answers = []
        descriptions = []
        
        for image_url in test_file:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": (
                            "You are an AI assistant that classifies images as advertisements. "
                            "Always respond in two clearly labeled sections:\n"
                            "Answer: <True/False>\n"
                            "Description: Two sentences to describe what’s happening in the photo"
                            f"Examples of advertisements include: {', '.join(advertisement_examples)}"
                        )}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_url},
                        {"type": "text", "text": "Is this image an advertisement? If unclear, say Answer: N/A."}
                    ]
                }
            ]
    
            output = pipe(text=messages, max_new_tokens=128)
            response = output[0]["generated_text"][-1]["content"]
    
            # Default values
            answer, rationale = "N/A", ""
    
            for line in response.splitlines():
                if line.startswith("Answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif line.startswith("Description:"):
                    rationale = line.split(":", 1)[1].strip()
    
            answers.append(answer)
            descriptions.append(rationale)
    
        # Final decision rule
        if "Yes" in answers:
            final_answer = "Yes"
        else:
            final_answer = "No"
    
        test_list.at[idx, 'is_image_ad'] = final_answer
        test_list.at[idx, 'image_Keywords'] = " | ".join(descriptions)

print(test_list)