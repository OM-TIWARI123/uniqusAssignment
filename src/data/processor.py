from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_filings(data_folder):
    """Extract text from PDF documents"""
    print("\n" + "="*60)
    print("STEP 3: Extracting text from PDF files...")
    print("="*60)
    
    loader = DirectoryLoader(
        str(data_folder),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    
    return documents


def chunk_documents(documents):
    """Split documents into chunks"""
    print("\n" + "="*60)
    print("STEP 4: Chunking documents...")
    print("="*60)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    return chunks