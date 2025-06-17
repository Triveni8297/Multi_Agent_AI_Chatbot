import os
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import GoogleSearchAPIWrapper
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "default-index")
# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# === Tool 1: FAISS Retrieval === #
@tool
def f1_faiss_search(query: str) -> str:
    """
    Search the unified FAISS index for a given query and return the most relevant documents.
    """
    print("🔧 Tool Invoked: f1_faiss_search")
    index_path = "faiss_index"  # Make sure this exists
    try:
        print(f"🔍 [FAISS] Searching FAISS index at '{index_path}' for query: '{query}'")
        faiss_index = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        results = faiss_index.similarity_search(query, k=3)
        print(f"✅ [FAISS] Found {len(results)} results.")
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"❌ [FAISS] Search failed: {str(e)}")
        return f"❌ FAISS search failed: {str(e)}"

# === Tool 2: Pinecone Retrieval === #
@tool
def f2_pinecone_search(query: str, namespace: str = "file1") -> str:
    """
    Search Pinecone index for a given query and return relevant documents from a given namespace.
    """
    print("🔧 Tool Invoked: f2_pinecone_search")
    try:

        pc = Pinecone(api_key=pinecone_api_key)

        # Load Pinecone vector store
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index_name,
            embedding=embedding_model,
            namespace=namespace
        )
        results = vectorstore.similarity_search(query, k=3)
        print(f"✅ [Pinecone] Found {len(results)} results.")
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"❌ [Pinecone] Search failed: {str(e)}")
        return f"❌ Pinecone search failed: {str(e)}"

# === Tool 3: Google Search === #
@tool
def f3_google_search(query: str) -> str:
    """
    Use Google Search API to search the internet and return top relevant results.
    """
    print("🔧 Tool Invoked: f3_google_search")
    try:
        print(f"🌐 [Google] Searching for query: '{query}'")
        search = GoogleSearchAPIWrapper()
        results = search.run(query)
        return results
    except Exception as e:
        print(f"❌ [Google] Search failed: {str(e)}")
        return f"❌ Google search failed: {str(e)}"


#EXAMPLE USAGE
if __name__ == "__main__":
    sample_query = "how can i contact them?"
    print(f"Running FAISS search for: {sample_query}")
    faiss_result = f1_faiss_search(sample_query)
    print(f"FAISS Result:\n{faiss_result}\n")

    print(f"Running Pinecone search for: {sample_query}")
    pinecone_result = f2_pinecone_search(sample_query)
    print(f"Pinecone Result:\n{pinecone_result}\n")

    print(f"Running Google search for: {sample_query}")
    google_result = f3_google_search(sample_query)
    print(f"Google Result:\n{google_result}\n")
    