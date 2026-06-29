# ================================================================
# WEEK 2 — Step 3: The RAG Agent
#
# This combines:
# - RETRIEVAL (find real chunks from the 10-K in ChromaDB)
# - GENERATION (LLM writes an answer using ONLY those real chunks)
#
# The key difference from Week 1: the agent CANNOT make up numbers.
# It can only use what's actually in the document.
# ================================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# ── Connect to the existing ChromaDB ─────────────────────────────
print("📂 Connecting to ChromaDB knowledge base...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="microsoft_10k"
)
print("✅ Connected")

# ── Set up the LLM ───────────────────────────────────────────────
llm = ChatOllama(model="llama3.2", temperature=0)

# ── THE RAG AGENT ────────────────────────────────────────────────
def rag_agent(question: str) -> str:
    print(f"\n🔍 RAG AGENT processing: {question}")

    # STEP 1: Retrieve relevant chunks from the real document
    print("   Retrieving relevant passages from 10-K...")
    docs = vectorstore.similarity_search(question, k=4)

    # STEP 2: Combine the chunks into context
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"   Found {len(docs)} relevant passages")

    # STEP 3: Ask the LLM to answer using ONLY the retrieved context
    prompt = f"""You are a financial analyst answering questions about Microsoft.

Use ONLY the information in the context below to answer.
If the answer is not in the context, say "This information is not in the filing."
Always cite specific numbers when available.

CONTEXT FROM MICROSOFT'S 10-K FILING:
{context}

QUESTION: {question}

ANSWER (using only the context above):"""

    response = llm.invoke(prompt)
    return response.content

# ── TEST IT ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("RAG AGENT — Answers from Microsoft's REAL 10-K")
    print("="*60)

    questions = [
        "What was Microsoft's total revenue growth?",
        "How did Microsoft's cloud business perform?",
        "What are the main risk factors Microsoft faces?",
    ]

    for q in questions:
        answer = rag_agent(q)
        print("\n" + "="*60)
        print(f"❓ {q}")
        print("="*60)
        print(answer)
        print()