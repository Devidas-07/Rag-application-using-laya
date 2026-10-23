import streamlit as st
import requests

API_URL = "http://localhost:8000/convesational-search"

st.title("Document Search")

query = st.text_input("Ask a question")

if st.button("Search") and query:
    response = requests.post(API_URL, json={"query": query})
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "answer" in data:
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Context Used")
            for source in data.get("sources", []):
                st.write(f"- **{source['id']}** — {source['title']}")
        else:
            st.subheader("Results")
            for item in data:
                st.write(f"- **{item['id']}** — {item['title']}")
    else:
        st.error(f"Request failed: {response.status_code}")
