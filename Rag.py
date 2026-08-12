import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

st.title("📚 Simple RAG Application")

# Knowledge base
documents = [
    "RAG stands for Retrieval Augmented Generation.",
    "RAG retrieves relevant information from documents before generating an answer.",
    "Cloudberry is a PostgreSQL-compatible database system.",
    "Python is a programming language commonly used for AI and machine learning.",
    "A vector database stores embeddings and helps find similar information."
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(documents)

# Create vector index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# User question
question = st.text_input("Ask a question:")

if question:

    # Convert question to embedding
    query_embedding = model.encode([question])

    # Retrieve relevant documents
    distances, indices = index.search(query_embedding, k=2)

    # Get context
    context = "\n".join(
        documents[i] for i in indices[0]
    )

    st.subheader("Retrieved Context")
    st.write(context)

    # Simple answer
    st.subheader("Answer")

    if "rag" in question.lower():
        st.write(
            "RAG means Retrieval Augmented Generation. "
            "It retrieves relevant information from documents "
            "and uses that information to answer questions."
        )
    elif "cloudberry" in question.lower():
        st.write(
            "Cloudberry is a PostgreSQL-compatible database system."
        )
    elif "python" in question.lower():
        st.write(
            "Python is a programming language commonly used "
            "for AI and machine learning."
        )
    else:
        st.write("I found this relevant information:")
        st.write(context)
