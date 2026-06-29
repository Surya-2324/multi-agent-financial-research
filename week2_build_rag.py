# ================================================================
# WEEK 2 — Step 2: Build the RAG knowledge base
#
# What this does:
# - Loads the downloaded 10-K text
# - Splits it into chunks (~1000 characters each)
# - Creates embeddings (turns text into numbers)
# - Stores everything in ChromaDB (vector database)
#
# After this, you can ask questions and get answers from the
# REAL document instead of the LLM guessing.
# ================================================================

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── STEP 1: Load the document ────────────────────────────────────
print("📂 Loading Microsoft 10-K...")
loader = TextLoader("data/microsoft_10k.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ Loaded {len(documents[0].page_content):,} characters")

# ── STEP 2: Split into chunks ────────────────────────────────────
# Why chunk? A 348K-char document is too big to search at once.
# We split it into small pieces so we can find the RELEVANT piece.
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # each chunk ~1000 characters
    chunk_overlap=200,    # overlap so we don't cut sentences in half
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ── STEP 3: Set up embeddings ────────────────────────────────────
# Embeddings turn text into numbers (vectors) so the computer can
# measure how similar two pieces of text are.
# This model runs locally — free, no API key.
print("\n🧮 Loading embedding model (first time downloads ~90MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embedding model ready")

# ── STEP 4: Store in ChromaDB ────────────────────────────────────
# ChromaDB is a vector database. It stores all the chunks as
# embeddings and lets us search them by meaning, not keywords.
print("\n💾 Building ChromaDB vector store (this takes 1-2 min)...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",   # saved to disk so we reuse it
    collection_name="microsoft_10k"
)
print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

# ── STEP 5: Test retrieval ───────────────────────────────────────
# Let's prove it works — ask a question and retrieve relevant chunks
print("\n" + "="*60)
print("TESTING RETRIEVAL")
print("="*60)

test_query = "What were Microsoft's main revenue sources and cloud business performance?"
print(f"\n🔍 Query: {test_query}")

results = vectorstore.similarity_search(test_query, k=3)

print(f"\n✅ Retrieved {len(results)} most relevant chunks:\n")
for i, doc in enumerate(results, 1):
    preview = doc.page_content[:300].replace("\n", " ")
    print(f"--- Chunk {i} ---")
    print(f"{preview}...")
    print()

print("="*60)
print("✅ RAG knowledge base ready!")
print("The vector store is saved in 'chroma_db/' folder.")
print("Next: build the RAG Agent that uses this to answer questions.")