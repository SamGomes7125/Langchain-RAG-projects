import langchain_helper as lch
import streamlit as st
import textwrap
st.title("Youtube Video Assistant")
youtube_url = st.text_input(label="Enter the YouTube video URL:", max_chars=50)
query = st.text_input(label="Enter your question about the video:", max_chars=50, key="query")
if st.button("Get Answer"):
    if youtube_url and query:
        db = lch.create_vectordb_from_youtube(youtube_url)
        response = lch.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.text(textwrap.fill(response, width=80))
    else:
        st.warning("Please enter correct YouTube video URL and your question.")
        