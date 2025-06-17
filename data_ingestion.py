import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pinecone import Pinecone, ServerlessSpec


# Load environment variables
load_dotenv()

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-west-2"
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "default-index")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# --- Utility Functions ---
def load_text_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def Gnerate_Embedd_Pincone(chunks: list) -> tuple:
    """Generate embeddings for chunks"""
    try:
        print("📡 Generating embeddings...")
        embeddings = embedding_model.embed_documents(chunks)
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise

def Gnerate_Faiss_Embedd(chunks: list, ):
    """Store chunks in FAISS using the embedding model and save locally"""
    try:
        print(f"💾 Creating FAISS index with {len(chunks)} chunks...")
        faiss_index = FAISS.from_texts(texts=chunks, embedding=embedding_model)
        save_path: str = "faiss_index"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        faiss_index.save_local(save_path)
        print(f"✅ FAISS index saved to: {save_path}")
        return faiss_index
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")
        raise



def store_data_pinecone(chunks: list, embeddings: list, namespace: str = "default_namespace"):
    """Store using native Pinecone SDK"""
    print(f"🚀 Storing in Pinecone namespace: {namespace} (Native method)...")

    # Check if index exists, create if it doesn't
    try:
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    except Exception as e:
        print(f"❌ Error listing indexes: {e}")
        existing_indexes = []

    if pinecone_index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {pinecone_index_name}")
        try:
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=pinecone_cloud,
                    region=pinecone_env
                )
            )
            print(f"✅ Created Pinecone index: {pinecone_index_name}")
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            raise
    
    # Get the index
    try:
        index = pc.Index(pinecone_index_name)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{namespace}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "namespace": namespace,
                    "chunk_id": i
                }
            })
        
        # Upsert vectors in batches
        batch_size = 10
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            print(f"📤 Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
        
        print(f"✅ Stored {len(chunks)} chunks in Pinecone index: {pinecone_index_name}, namespace: {namespace}")
        
    except Exception as e:
        print(f"❌ Error storing in Pinecone: {e}")
        raise



def process_txt_file(file_path: str):
    """Process .txt file and store in FAISS"""
    try:
        print(f"\n📄 Processing TXT file: {file_path}")
        
        # Load and chunk the text
        text_content = load_text_file(file_path)
        chunks = chunk_text(text_content)
        print(f"📝 Text split into {len(chunks)} chunks")
        
        # Generate embeddings and store in FAISS
        faiss_index = Gnerate_Faiss_Embedd(chunks)
        return  faiss_index
    except Exception as e:
        print(f"❌ Error processing TXT file: {e}")
        raise


def process_md_file(file_path: str):
    """Process .md file and store in Pinecone"""
    try:
        print(f"\n📄 Processing MD file: {file_path}")
        
        # Load and chunk the text
        text_content = load_text_file(file_path)
        chunks = chunk_text(text_content)
        print(f"📝 Text split into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = Gnerate_Embedd_Pincone(chunks)
        # Store in Pinecone using native method only
        pinecone_db = store_data_pinecone(chunks, embeddings)
        return pinecone_db
    except Exception as e:
        print(f"❌ Error processing MD file: {e}")
        raise


def main_code(txt_file_path: str, md_file_path: str):
    """
    Main function to process files and store embeddings.
    
    Args:
        txt_file_path (str): Path to the .txt file to be processed.
        md_file_path (str): Path to the .md file to be processed.
    
    Returns:
        dict: Summary of processing results.
    """
    try:
        faiss_index = process_txt_file(txt_file_path)
        print(f"✅ TXT file processed and stored in FAISS index.")
        pinecone_db = process_md_file(md_file_path)
        print(f"✅ MD file processed and stored in Pinecone index.")
        return "Successfully processed both files and stored embeddings."
    except Exception as e:
        print(f"❌ Error in main processing: {e}")
        return {}




# example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    txt_file_path = "crawled_data.txt"  # Your .txt file
    md_file_path = "parsed_case_2.md"    # Your .md file

    # Process both files
    results = main_code(
        txt_file_path=txt_file_path,
        md_file_path=md_file_path,
    )
    
    print("\n📊 Processing Summary:")
    print(results)