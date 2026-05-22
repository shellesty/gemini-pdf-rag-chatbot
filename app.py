import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Streamlit UI
st.set_page_config(page_title="PDF RAG Chatbot")

st.title("📄 PDF RAG Chatbot")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    extracted_text = ""

    for page in pdf_reader.pages:

        extracted_text += page.extract_text()

    st.subheader("Extracted Text Preview")

    st.write(extracted_text[:3000])

    # Chunking function
    def chunk_text(text, chunk_size=500):

        chunks = []

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            chunks.append(chunk)

        return chunks


    # Create chunks
    text_chunks = chunk_text(extracted_text)

    # Display chunk information
    st.subheader("Text Chunks")

    st.write(f"Total Chunks Created: {len(text_chunks)}")

    # Preview first chunk
    st.write(text_chunks[0])