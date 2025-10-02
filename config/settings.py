import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "Data_Scope"

# Company configurations
COMPANIES = {
    "GOOGL": "0001652044",
    "MSFT": "0000789019",
    "NVDA": "0001045810"
}

YEARS = [2022, 2023, 2024]

# Chroma settings
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "sec_filings"

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 500
MAX_CHUNKS = 5000

# Retrieval settings
RETRIEVAL_K = 3

# LLM settings
LLM_MODEL = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Downloader settings
DOWNLOADER_COMPANY = "Uniqus Assignment"
DOWNLOADER_EMAIL = "omt887@gmail.com"