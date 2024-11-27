# import streamlit as st
# from langchain_community.tools import DuckDuckGoSearchRun

# # Initialize DuckDuckGoSearchRun
# search = DuckDuckGoSearchRun()

# # Streamlit app
# st.title("Search AI Companies in Hyderabad")

# # Input field for search query
# query = st.text_input("Enter your search query:", "AI Planet Company in Hyderabad")

# # Button to trigger the search
# if st.button("Search"):
#     with st.spinner("Searching..."):
#         try:
#             # Invoke the search
#             results = search.invoke(query)
#             # Display results
#             st.subheader("Search Results:")
#             st.write(results)
#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# # Footer
# st.caption("Powered by DuckDuckGo and LangChain Community Tools")


import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from llama_index import GPTSimpleVectorIndex, Document
from huggingface_hub import InferenceClient

# Initialize DuckDuckGoSearchRun and Hugging Face client with API key
search = DuckDuckGoSearchRun()
client = InferenceClient(api_key="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")

# Initialize LlamaIndex instances
research_index = None
use_case_index = None

# Streamlit App
st.title("AI Use Case Generator with Hugging Face LLM")

# Input fields for industry and company
industry = st.text_input("Enter the Industry (e.g., Retail, Automotive):")
company = st.text_input("Enter the Company Name:")
query = f"{company} in {industry}"

# Agent Workflow
def research_industry_and_company(query):
    results = search.invoke(query)
    documents = [Document(text=result) for result in results]
    global research_index
    research_index = GPTSimpleVectorIndex.from_documents(documents)
    return results

def generate_use_cases_with_hf(industry):
    # Prepare input for the Hugging Face model
    messages = [
        {
            "role": "user",
            "content": f"Suggest relevant AI and GenAI use cases for the {industry} industry, focusing on operations, supply chain, and customer experience."
        }
    ]
    
    # Call the Hugging Face Inference API
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
    
    # Index the generated use cases
    global use_case_index
    use_case_index = GPTSimpleVectorIndex.from_documents([Document(text=response)])
    return response

def search_index(index, query):
    if index is None:
        return "No data available to search."
    response = index.query(query)
    return response.response

# Streamlit Workflow
if st.button("Generate"):
    with st.spinner("Processing..."):
        try:
            # Research Phase
            st.subheader("Industry and Company Research")
            research_results = research_industry_and_company(query)
            st.write(research_results)
            
            # Use Case Generation
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
            index = research_index if index_type == "Research" else use_case_index
            search_results = search_index(index, search_query)
            st.write(search_results)
        except Exception as e:
            st.error(f"An error occurred: {e}")
