import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import InferenceClient

# Initialize DuckDuckGoSearchRun and Hugging Face client
search = DuckDuckGoSearchRun()
client = InferenceClient(api_key="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")

# Dictionary to store indexed research and use cases
research_index = {}
use_case_index = {}

# Streamlit App
st.title("AI Use Case Generator with Hugging Face LLM")

# Input fields for industry and company
industry = st.text_input("Enter the Industry (e.g., Retail, Automotive):")
company = st.text_input("Enter the Company Name:")
query = f"{company} in {industry}"

# Agent Workflow
def research_industry_and_company(query):
    results = search.invoke(query)
    research_index["results"] = results  # Storing research results in a simple dictionary
    return results

def generate_use_cases_with_hf(industry):
    messages = [
        {
            "role": "user",
            "content": f"Suggest relevant AI and GenAI use cases for the {industry} industry, focusing on operations, supply chain, and customer experience."
        }
    ]
    
    response = ""
    with st.spinner("Generating use cases with Hugging Face LLM..."):
        stream = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=500,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            response += delta
            st.write(delta, end="")
    
    use_case_index["use_cases"] = response  # Store the use case response in the dictionary
    return response

def search_index(index, query):
    if index is None or query not in index:
        return "No data available to search."
    return index.get(query, "Data not found.")

# Streamlit Workflow
if st.button("Generate"):
    with st.spinner("Processing..."):
        try:
            st.subheader("Industry and Company Research")
            research_results = research_industry_and_company(query)
            st.write(research_results)
            
            st.subheader("AI Use Cases")
            use_cases = generate_use_cases_with_hf(industry)
            st.write(use_cases)
            
            st.success("Data Indexed Successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Search functionality
st.subheader("Search Indexed Data")
index_type = st.radio("Select Index to Search:", ("Research", "Use Cases"))
search_query = st.text_input("Enter your search query:")
if st.button("Search Index"):
    with st.spinner("Searching..."):
        try:
            # Select the correct index based on user selection
            index = research_index if index_type == "Research" else use_case_index
            search_results = search_index(index, search_query)
            st.write(search_results)
        except Exception as e:
            st.error(f"An error occurred: {e}")
