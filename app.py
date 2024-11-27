import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

# Streamlit app
st.title("Search AI Companies in Hyderabad")

# Input field for search query
query = st.text_input("Enter your search query:", "AI Planet Company in Hyderabad")

# Button to trigger the search
if st.button("Search"):
    with st.spinner("Searching..."):
        try:
            # Invoke the search
            results = search.invoke(query)
            # Display results
            st.subheader("Search Results:")
            st.write(results)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.caption("Powered by DuckDuckGo and LangChain Community Tools")
