import os
os.system('pip install langchain')

import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

st.title("News Analysis Chatbot")

url1=st.text_input("Enter URL", key='url1')
url2=st.text_input("Enter URL", key='url2')

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

if url1 and url2:
    loader = UnstructuredURLLoader(urls=[url1, url2])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(docs, embeddings)

    from langchain.llms import HuggingFaceHub
    api_key='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'

    llm = HuggingFaceHub(
        huggingfacehub_api_token=api_key,
        repo_id="google/flan-t5-large",
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.85,
            "max_length": 150  # Increase max_length for longer outputs
        }
    )

    def generate_response(query):
        docs=docsearch.similarity_search(query, k=2)
        response2=llm(f'Tell me {query} on the basis of the context {docs[1].page_content}')
        response1=llm(f'Tell me {query} on the basis of the context {docs[0].page_content}')
        combined_response=llm(f'Tell me {query} on the basis of the context {docs[0].page_content} and {docs[1].page_content}')
        return response1, response2, combined_response
    
    query=st.text_input("Enter your query", key='query')
    if st.button("Submit"):
        response1, response2, combined_response = generate_response(query)
        st.write("Response 1:", response1)
        st.write("Response 2:", response2)
        st.write("Combined Response:", combined_response)
