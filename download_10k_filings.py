import os
from sec_edgar_downloader import Downloader

# Create a folder for the downloaded filings
data_folder = "Data_Scope"
os.makedirs(data_folder, exist_ok=True)

# Initialize the downloader with email address (required by SEC)
# Using a placeholder email; you should replace this with your actual email
dl = Downloader("Uniqus Assignment", "omt887@gmail.com", data_folder)

# Company CIK codes and tickers
companies = {
    "GOOGL": "0001652044",
    "MSFT": "0000789019",
    "NVDA": "0001045810"
}

# Years to download
years = [2022, 2023, 2024]

# Download 10-K filings for each company and year
total_filings = 0
for ticker, cik in companies.items():
    print(f"Downloading 10-K filings for {ticker} (CIK: {cik})...")
    for year in years:
        try:
            # Download 10-K filing for the specific year
            dl.get("10-K", cik, after=str(year)+"-01-01", before=str(year)+"-12-31", download_details=True)
            print(f"  Successfully downloaded {ticker} 10-K for {year}")
            total_filings += 1
        except Exception as e:
            print(f"  Failed to download {ticker} 10-K for {year}: {str(e)}")

print(f"\nDownload complete! Total filings downloaded: {total_filings}")