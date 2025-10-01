import os
import pdfkit

base_dir = "Data_Scope"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".html"):
            html_path = os.path.join(root, file)
            pdf_path = os.path.splitext(html_path)[0] + ".pdf"  # same name, .pdf
            
            try:
                pdfkit.from_file(html_path, pdf_path)
                print(f"Converted: {html_path} → {pdf_path}")
            except Exception as e:
                print(f"Failed for {html_path}: {e}")
