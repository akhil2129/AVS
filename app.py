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
    # Preparing input for the LLM
    messages = [
        {
            "role": "user",
            "content": f"Suggest relevant AI and GenAI use cases for the {industry} industry, focusing on operations, supply chain, and customer experience. Highlight actionable insights and potential challenges."
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

def format_search_results(raw_results):
    # Process raw search results using the LLM for better formatting
    if not raw_results:
        return "No relevant information found for your query."
    
    # Sample formatted content
    formatted_output = f"""
    ## Search Results for Query:
    {query}
    
    **Top Findings**:
    - **Key Industry Trends**: [e.g., "Emphasis on AI-driven customer experience", "Integration of predictive analytics in supply chain operations"]
    - **Actionable Insights**:
        - **Implement real-time data analytics**: Can help in monitoring customer behavior and improving operations.
        - **Use AI for predictive maintenance**: Enhances supply chain efficiency and reduces costs.
    - **Challenges**:
        - **Data privacy and security concerns**: Handling customer data responsibly is crucial.
        - **Integration with legacy systems**: Ensuring compatibility with existing software is necessary.
    - **Potential Solutions**:
        - **Use API-driven architecture**: Streamlined integration between new AI applications and legacy systems.
        - **Leverage open-source AI models**: Can reduce costs and improve model adaptability.
    """
    
    return formatted_output

def search_index(index, query):
    if index is None or query not in index:
        return "No data available to search."
    results = index.get(query, None)
    if results:
        return format_search_results(results)
    else:
        return "No relevant information found for your query."

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
            st.markdown(search_results, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
