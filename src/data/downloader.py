import os
from sec_edgar_downloader import Downloader
from config.settings import (
    COMPANIES, YEARS, DATA_FOLDER,
    DOWNLOADER_COMPANY, DOWNLOADER_EMAIL
)


def download_filings():
    """Download 10-K filings for specified companies and years"""
    print("\n" + "="*60)
    print("STEP 1: Downloading 10-K filings...")
    print("="*60)
    
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    dl = Downloader(DOWNLOADER_COMPANY, DOWNLOADER_EMAIL, str(DATA_FOLDER))
    
    total_filings = 0
    for ticker, cik in COMPANIES.items():
        print(f"\n📥 Downloading filings for {ticker} (CIK: {cik})...")
        for year in YEARS:
            try:
                dl.get(
                    "10-K", 
                    cik, 
                    after=f"{year}-01-01",
                    before=f"{year}-12-31",
                    download_details=True
                )
                print(f"  ✓ Successfully downloaded {ticker} 10-K for {year}")
                total_filings += 1
            except Exception as e:
                print(f"  ✗ Failed to download {ticker} 10-K for {year}: {str(e)}")
    
    print(f"\n✅ Download complete! Total filings downloaded: {total_filings}")
    return DATA_FOLDER