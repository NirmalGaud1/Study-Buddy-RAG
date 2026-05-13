import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np

st.set_page_config(page_title="Study Buddy RAG")

st.title("📚 Study Buddy RAG")
st.write("Upload your textbook, notes, or study material and ask questions from it.")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_resource
def load_llm():
    return genai.GenerativeModel("models/gemini-2.0-flash-lite")

llm = load_llm()

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def chunk_text(text, chunk_size=400, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

@st.cache_resource
def create_vector_store(chunks):

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

    return index

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        full_text = extract_text(uploaded_file)

        chunks = chunk_text(full_text)

        vector_store = create_vector_store(chunks)

    st.success(f"✅ PDF processed successfully! {len(chunks)} chunks created.")

    question = st.text_input("Ask a question from your document")

    if st.button("Get Answer"):

        if question.strip() == "":

            st.warning("Please enter a question.")

        else:

            query_embedding = embedding_model.encode([question])

            query_embedding = np.array(query_embedding).astype("float32")

            distances, indices = vector_store.search(query_embedding, k=3)

            retrieved_chunks = [chunks[i] for i in indices[0]]

            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
You are a Study Buddy assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, reply with:

"I could not find the answer in the uploaded document."

Keep answers concise and student-friendly.

Context:
{context}

Question:
{question}

Answer:
"""

            try:

                with st.spinner("Generating Answer..."):

                    response = llm.generate_content(prompt)

                    st.subheader("📖 Answer")

                    st.write(response.text)

                with st.expander("📄 Source Chunks Used"):

                    for i, chunk in enumerate(retrieved_chunks):

                        st.markdown(f"### Chunk {i+1}")

                        st.write(chunk)

            except Exception as e:

                st.error(f"Error: {e}")
