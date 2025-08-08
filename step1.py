import zipfile
import os

# Define the path to the uploaded zip file and the extraction directory
zip_path = r"C:\Users\saadk\Desktop\Week4 intern\archive.zip"

extract_dir = r"C:\Users\saadk\Desktop\Week4 intern"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# List the extracted files
extracted_files = os.listdir(extract_dir)
extracted_files
