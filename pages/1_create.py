import streamlit as st
from utils import generate_complete_dataset
import pandas as pd
from utils import json_to_csv
import json
import os

st.set_page_config(page_title="GENIE - Create Dataset", page_icon="📝")

# Function to add another row for the fields
def add_field(field_count):
    field_label = st.text_input(f'Field {field_count} Label', key=f'field_label_{field_count}')
    field_type = st.text_input(f'Field {field_count} Data Type', key=f'field_type_{field_count}')
    return field_label, field_type

# Main function for the Streamlit app
def main():
    st.title("Dataset Field Specification")

    # Initialize field count if not in session state
    if 'field_count' not in st.session_state:
        st.session_state.field_count = 1

    # Form for the dataset fields
    with st.form(key='fields_form'):
        # Title input
        title = st.text_input("Title")

        fields = []
        # Display current fields
        for i in range(1, st.session_state.field_count + 1):
            field_label, field_type = add_field(i)
            if field_label and field_type:
                fields.append((field_label, field_type))

        # Button to add more fields
        if st.form_submit_button("Add More Fields") and st.session_state.field_count < 15:
            st.session_state.field_count += 1
            st.rerun()

        # Additional info input
        additional_info = st.text_area("Additional Info", 
                                       placeholder="Please add anything else you want to add about the dataset, any guidance or rules, or paste any examples.")

        # Dropdown for number of rows in the dataset
        number_of_rows = st.selectbox("Number of Rows in Dataset", options=[str(i) for i in range(10, 101, 10)])
        
        # Save the form data and process the dataset
        if st.form_submit_button("Generate Dataset", use_container_width=True):
            st.session_state['title'] = title
            st.session_state['additional_info'] = additional_info

            # Prepare the dictionary-like string
            fields_dict_str = str({label: dtype for label, dtype in fields})

            st.session_state['fields_dict_str'] = fields_dict_str
            st.session_state['number_of_rows'] = int(number_of_rows)

            # Generate the dataset
            complete_dataset = generate_complete_dataset(
                title=st.session_state['title'],
                number=st.session_state['number_of_rows'],
                fields=st.session_state['fields_dict_str'],
                additional_info=st.session_state['additional_info']
            )

            st.success("Dataset generated successfully!")

            # Save the dataset to a JSON file in a temporary location
            with open(f"{st.session_state['title']}.json", "w") as f:
                json.dump(complete_dataset, f, indent=4)

            # Convert the dataset to CSV format
            csv_data = json_to_csv(complete_dataset)

            # Save the dataset to a CSV file in a temporary location
            with open(f"{st.session_state['title']}.csv", "w", newline="") as f:
                f.write(csv_data)

            # Make the download button visible
            st.session_state['dataset_ready'] = True

    # Show download button if dataset is ready
    if 'dataset_ready' in st.session_state and st.session_state['dataset_ready']:
        with open(f"{st.session_state['title']}.json", "rb") as f:
            st.download_button(
                label="Download JSON file",
                data=f,
                file_name=f"{st.session_state['title']}.json",
                mime="application/json"
            )
            
    if 'dataset_ready' in st.session_state and st.session_state['dataset_ready']:
        with open(f"{st.session_state['title']}.csv", "rb") as f:
            st.download_button(
                label="Download CSV file",
                data=f,
                file_name=f"{st.session_state['title']}.csv",
                mime="text/csv"
            )

# Run the app
if __name__ == "__main__":
    main()
