import os
# from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
import json
import re
import random
import csv
import io
import streamlit as st

# load_dotenv()



URL = "https://us-south.ml.cloud.ibm.com"
# IBM_API_KEY = os.getenv("IBM_API_KEY")
# project_id = os.getenv("PROJECT_ID")

IBM_API_KEY = st.secrets["IBM_API_KEY"]
project_id = st.secrets["PROJECT_ID"]

def get_credentials():
    return {
        "url": URL,
        "apikey": IBM_API_KEY
    }

def get_response(title, fields, additional_info):
    model_id = "ibm/granite-13b-chat-v2"

    seed = random.randint(0, 100000)

    parameters = {
        "decoding_method": "sample",
        "max_new_tokens": 5000,
        "random_seed": seed,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1,
        "repetition_penalty": 1
    }

    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=get_credentials(),
        project_id=project_id,
    )

    # Construct the prompt
    prompt_input = f"""Generate a synthetic dataset for the {title} dataset whose column fields are pasted below. 
                    Adhere to the data type associated with each field. Answer purely in JSON format and no other text. 
                    Do not generate unrealistic or made-up data - stick to real-world factual concepts. Do not leave any field blank, make sure 
                    each field has a valid value. Create 10 rows of data. Your response should contain no other text EXCEPT the dataset. 

                    FIELDS and their DATATYPES:

                    {fields}

                    ADDITIONAL INFO: {additional_info}

                    Your response should follow this format EXACTLY:
                    [{fields}]"""
    
    generated_response = model.generate_text(prompt=prompt_input, guardrails=True)
    print(generated_response)
    return generated_response

def extract_json_from_response(response_text):
    # Find the first occurrence of "[" and the last occurrence of "]"
    start_idx = response_text.find("[")
    end_idx = response_text.rfind("]") + 1  # Include the closing bracket

    if start_idx != -1 and end_idx != -1:
        json_str = response_text[start_idx:end_idx]

        # Check if single quotes are used instead of double quotes
        if "'" in json_str and not '"' in json_str:
            json_str = json_str.replace("'", '"')

        # Try parsing the JSON string
        try:
            # Use the json.loads() method to parse the string
            dataset = json.loads(json_str)
            return dataset
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            print("Attempting to clean the JSON string...")

            # Try to clean up the string
            json_str = re.sub(r"(\w+):", r'"\1":', json_str)  # Add quotes around keys if missing
            json_str = re.sub(r':\s*([^{\["\']+)', r': "\1"', json_str)  # Add quotes around values if missing

            try:
                dataset = json.loads(json_str)
                return dataset
            except json.JSONDecodeError as e:
                print(f"Second attempt at JSON decoding failed: {e}")
                return None
    else:
        print("No JSON array found in the response.")
        return None

def generate_complete_dataset(title, number, fields, additional_info):
    all_data = []

    # Determine how many iterations of 10 we need to run
    iterations = number // 10

    for i in range(iterations):
        # Generate the dataset batch
        response = get_response(title, fields, additional_info)
        batch_data = extract_json_from_response(response)
        print(f"Batch {i+1} generated successfully.")
        print(batch_data)

        if batch_data:
            all_data.extend(batch_data)

    # Save the entire dataset as a JSON file
    with open(f"{title}.json", 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

    print(f"Complete dataset successfully generated and saved to complete_dataset.json")
    return all_data

def json_to_csv(json_data):
    # Create an in-memory file
    output = io.StringIO()
    
    # Extract the keys (column names) from the first dictionary in the list
    keys = json_data[0].keys()

    # Create a CSV writer object
    dict_writer = csv.DictWriter(output, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(json_data)

    # Get the CSV content as a string
    csv_content = output.getvalue()
    output.close()

    return csv_content
