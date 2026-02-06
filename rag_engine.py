import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
CHROMA_PATH = "./policy_db"
os.makedirs(CHROMA_PATH, exist_ok=True)

# 1. Initialize Embeddings & Splitters
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# We use one splitter to create the 'Parent' (Context) 
# and another to create the 'Child' (Searchable)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 2. Setup Persistent Chroma
vectorstore = Chroma(
    collection_name="insurance_policy",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

def add_policy_to_system(file_path: str):
    """
    Ingests a PDF by creating child chunks that carry 
    their parent's full context in metadata.
    """
    loader = PyPDFLoader(file_path)
    full_docs = loader.load()
    
    final_chunks = []
    
    # Process each page
    for doc in full_docs:
        # Create Parents
        parents = parent_splitter.split_documents([doc])
        
        for p_doc in parents:
            # Create Children from this parent
            children = child_splitter.split_documents([p_doc])
            
            for c_doc in children:
                # 🧠 THE TRICK: Attach the parent text to the child's metadata
                c_doc.metadata["parent_context"] = p_doc.page_content
                c_doc.metadata["source_file"] = os.path.basename(file_path)
                final_chunks.append(c_doc)
    
    # Add to persistent store
    vectorstore.add_documents(final_chunks)
    return True

def get_policy_context(query: str):
    """
    Searches child chunks but returns the full Parent Context.
    This gives the LLM high-precision results + full legal context.
    """
    results = vectorstore.similarity_search(query, k=3)
    
    formatted_context = ""
    for doc in results:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "??")
        # Pull the 'Parent' text we hid in the metadata earlier
        parent_text = doc.metadata.get("parent_context", doc.page_content)
        
        formatted_context += f"\n--- FROM: {source} (Page {page}) ---\n{parent_text}\n"
        
    return formatted_context