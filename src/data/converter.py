import os
import pdfkit
from pathlib import Path


def convert_html_to_pdf(data_folder):
    """Convert all HTML files in the data folder to PDF"""
    print("\n" + "="*60)
    print("STEP 2: Converting HTML files to PDF...")
    print("="*60)
    
    converted_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".html"):
                html_path = os.path.join(root, file)
                pdf_path = os.path.splitext(html_path)[0] + ".pdf"
                
                # Skip if PDF already exists
                if os.path.exists(pdf_path):
                    print(f"  ⊙ PDF already exists: {pdf_path}")
                    continue
                
                try:
                    pdfkit.from_file(html_path, pdf_path)
                    print(f"  ✓ Converted: {os.path.basename(html_path)} → {os.path.basename(pdf_path)}")
                    converted_count += 1
                except Exception as e:
                    print(f"  ✗ Failed: {os.path.basename(html_path)} - {e}")
                    failed_count += 1
    
    print(f"\n✅ Conversion complete! Converted: {converted_count}, Failed: {failed_count}")
    return converted_count