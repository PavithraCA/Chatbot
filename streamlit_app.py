import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Set up Langchain and OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Set up text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=3000,
    chunk_overlap=00
)

# Load text data using TextLoader
loader = TextLoader("/Users/abhay/Downloads/DEV/stream/Dronealexa.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# Create FAISS index
db = FAISS.from_documents(docs, OpenAIEmbeddings())

# Function to summarize text using pipeline
def summarize_text(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, length_penalty=2.0)[0]['summary_text']
    return summary

# Streamlit app
st.title(" Patent Chatbot ")

# Input text from user
user_input = st.text_area("Shoot Your Question:")

# Perform similarity search
if st.button("Enter"):
    # Perform similarity search in the database
    search_result = db.similarity_search(user_input)
    
    if search_result:
        # Summarize the relevant content
        summary_result = summarize_text(search_result[0].page_content)
        st.subheader("Summary of the  content :")
        st.write(summary_result)
    
        # Display the most relevant content
        st.subheader("Relevant Content:")
        st.write(search_result[0].page_content)
    else:
        st.warning("No relevant content found.")

