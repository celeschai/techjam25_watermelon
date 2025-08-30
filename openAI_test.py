import csv
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def format_row_for_openai(row):
    """Format a CSV row for OpenAI analysis, handling empty fields properly."""
    formatted_data = {
        "time": row.get('time', ''),
        "text": row.get('text', ''),
        "pics": row.get('pics', ''),
        "resp": row.get('resp', ''),
        "name_of_business": row.get('name_of_business', ''),
        "description": row.get('description', ''),
        "category": row.get('category', ''),
        "url": row.get('url', '')
    }
    
    # Convert empty strings to "N/A" for better OpenAI understanding
    for key, value in formatted_data.items():
        if value == '' or value is None:
            formatted_data[key] = "N/A"
    
    return formatted_data

def analyze_with_openai(formatted_data):
    """Send data to OpenAI for analysis and return structured response."""
    
    prompt = f"""
    Analyze the following business review data and determine policy violations and relevance:

    Business: {formatted_data['name_of_business']}
    Category: {formatted_data['category']}
    Description: {formatted_data['description']}
    Review Text: {formatted_data['text']}
    Images: {formatted_data['pics']}
    Response: {formatted_data['resp']}

    Please analyze and respond in the following JSON format:
    {{
        "text_violations": {{
            "is_Text_Ad": false,
            "is_Text_Rant": false,
            "is_Text_Irrelevant": false
        }},
        "image_violations": {{
            "is_Image_Ad": false,
            "is_Image_Irrelevant": false
        }},
        "relevance_assessment": "helpful",
        "rationale": "Brief explanation of the assessment"
    }}

    Guidelines:
    - Text Ad: Contains promotional content, marketing language, or advertising
    - Text Rant: Contains excessive negative language, personal attacks, or unreasonable complaints
    - Text Irrelevant: Content unrelated to the business or service
    - Image Ad: Contains promotional imagery, logos, or marketing content
    - Image Irrelevant: Images not related to the business, service, or review
    - Relevance: "Irrelevant", "not helpful", "helpful", or "very helpful" based on business context
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert content moderator analyzing business reviews for policy violations and relevance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Extract the response content
        response_text = response.choices[0].message.content
        
        # Try to parse JSON from the response
        try:
            # Find JSON content in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            print(f"Warning: Could not parse JSON response for row. Raw response: {response_text}")
            return create_fallback_response()
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return create_fallback_response()

def create_fallback_response():
    """Create a fallback response when OpenAI fails."""
    return {
        "text_violations": {
            "is_Text_Ad": False,
            "is_Text_Rant": False,
            "is_Text_Irrelevant": False
        },
        "image_violations": {
            "is_Image_Ad": False,
            "is_Image_Irrelevant": False
        },
        "relevance_assessment": "not helpful",
        "rationale": "Analysis failed - using fallback values"
    }

def main():
    """Main function to process the CSV file."""
    input_file = 'sample_input.csv'
    output_file = 'output.csv'
    
    results = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                print(f"\nProcessing row {i+1}...")
                
                # Format the row data
                formatted_data = format_row_for_openai(row)
                
                # Analyze with OpenAI
                analysis = analyze_with_openai(formatted_data)
                
                # Print rationale to screen
                print(f"Rationale: {analysis.get('rationale', 'No rationale provided')}")
                
                # Prepare output row
                output_row = {
                    'is_Text_Ad': analysis['text_violations']['is_Text_Ad'],
                    'is_Text_Rant': analysis['text_violations']['is_Text_Rant'],
                    'is_Text_Irrelevant': analysis['text_violations']['is_Text_Irrelevant'],
                    'is_Image_Ad': analysis['image_violations']['is_Image_Ad'],
                    'is_Image_Irrelevant': analysis['image_violations']['is_Image_Irrelevant'],
                    'relevancy_final': analysis['relevance_assessment']
                }
                
                results.append(output_row)
                
                # Add a small delay to avoid rate limiting
                import time
                time.sleep(0.5)
        
        # Write results to output CSV
        if results:
            fieldnames = ['is_Text_Ad', 'is_Text_Rant', 'is_Text_Irrelevant', 'is_Image_Ad', 'is_Image_Irrelevant', 'relevancy_final']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\nAnalysis complete! Results saved to {output_file}")
            print(f"Processed {len(results)} rows successfully.")
        else:
            print("No results to write.")
            
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()