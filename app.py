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
    page_title="Multi-PDF Conversational RAG Chatbot",
    page_icon="📚"
)

# App title
st.title("📚 Multi-PDF Conversational RAG Chatbot")

st.write("Upload multiple PDFs and chat with them.")

# Initialize chat history
if "messages" not in st.session_state:

    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# Chunking function
def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(
            {
                "text": chunk
            }
        )

    return chunks

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    # Store all extracted text
    extracted_text = ""

    # Store chunks with metadata
    document_chunks = []

    # Process each uploaded PDF
    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        pdf_text = ""

        # Extract text from pages
        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:

                pdf_text += text

        # Create chunks for this document
        pdf_chunks = chunk_text(pdf_text)

        # Add source metadata
        for chunk in pdf_chunks:

            document_chunks.append(
                {
                    "source": uploaded_file.name,
                    "text": chunk["text"]
                }
            )

        # Add separator between PDFs
        extracted_text += pdf_text + "\n\n"

    # Display extracted text preview
    st.subheader("Extracted Text Preview")

    st.write(extracted_text[:3000])

    # Extract chunk texts
    text_chunks = [
        chunk["text"]
        for chunk in document_chunks
    ]

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
        "Ask a question about the PDFs"
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

        # Convert question into embedding
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
                document_chunks[idx]["text"]
            )

        # Display retrieved chunks
        st.subheader("Retrieved Chunks")

        for idx in indices[0]:

            st.write(
                f"📄 Source: {document_chunks[idx]['source']}"
            )

            st.write(
                document_chunks[idx]["text"]
            )

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

        # Generate Gemini streaming response
        try:

            # Create assistant chat container
            with st.chat_message("assistant"):

                response_placeholder = st.empty()

                full_response = ""

                # Stream response
                response_stream = client.models.generate_content_stream(
                    model="gemini-3-flash-preview",
                    contents=prompt
                )

                for chunk in response_stream:

                    if chunk.text:

                        full_response += chunk.text

                        response_placeholder.markdown(full_response)

            # Store assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )

        except Exception as e:

            st.error(f"Error: {e}")