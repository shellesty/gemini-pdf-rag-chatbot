import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Streamlit page config
st.set_page_config(
    page_title="Conversational PDF RAG Chatbot",
    page_icon="📄"
)

# App title
st.title("📄 Conversational PDF RAG Chatbot")

st.write("Upload a PDF and chat with it.")

# Initialize chat history
if "messages" not in st.session_state:

    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    extracted_text = ""

    # Extract text from each page
    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:
            extracted_text += text

    # Display extracted text preview
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

    # Generate embeddings
    embeddings = embedding_model.encode(text_chunks)

    # Display embedding information
    st.subheader("Embeddings")

    st.write(f"Total Embeddings Created: {len(embeddings)}")

    st.write(f"Embedding Dimension: {len(embeddings[0])}")

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    # Display FAISS information
    st.subheader("FAISS Vector Store")

    st.write(f"Total vectors stored: {index.ntotal}")

    # User chat input
    user_question = st.chat_input(
        "Ask a question about the PDF"
    )

    if user_question:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        # Display user message
        with st.chat_message("user"):

            st.markdown(user_question)

        # Convert user question into embedding
        question_embedding = embedding_model.encode(
            [user_question]
        )

        # Convert to numpy array
        question_embedding = np.array(question_embedding)

        # Search FAISS index
        distances, indices = index.search(
            question_embedding,
            k=3
        )

        # Retrieve relevant chunks
        retrieved_chunks = []

        for idx in indices[0]:

            retrieved_chunks.append(
                text_chunks[idx]
            )

        # Display retrieved chunks
        st.subheader("Retrieved Chunks")

        for chunk in retrieved_chunks:

            st.write(chunk)

            st.write("------")

        # Combine retrieved chunks
        context = "\n\n".join(retrieved_chunks)

        # Build conversation history
        conversation_history = ""

        for message in st.session_state.messages:

            role = message["role"]

            content = message["content"]

            conversation_history += f"{role}: {content}\n"

        # Create prompt
        prompt = f"""
        You are a helpful AI assistant.

        Answer the question based ONLY on the provided context and conversation history.

        Context:
        {context}

        Conversation History:
        {conversation_history}

        Current Question:
        {user_question}
        """

        # Generate Gemini response
        try:

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            # Display assistant response
            with st.chat_message("assistant"):

                st.markdown(response.text)

            # Store assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.text
                }
            )

        except Exception as e:

            st.error(f"Error: {e}")