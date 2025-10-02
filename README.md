# RAG Agent for 10-K Filings

A RAG-based agent for querying 10-K filings of Google, Microsoft, and NVIDIA.

## Prerequisites

- Docker
- Python 3.x
- wkhtmltopdf

### Install wkhtmltopdf

**Linux:**
```bash
sudo apt-get install wkhtmltopdf
```

**MacOS:**
```bash
brew install wkhtmltopdf
```

**Windows:**
Download and install from: https://wkhtmltopdf.org/downloads.html

## Setup Instructions

1. **Start ChromaDB using Docker**
   ```bash
   docker run -d --rm --name chromadb -p 8000:8000 -v ./chroma-data:/data chromadb/chroma
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root and add your Gemini API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Notes

- The agent downloads 10-K filings using the SEC Edgar Python library
- HTML files are converted to PDF using pdfkit (requires wkhtmltopdf)
