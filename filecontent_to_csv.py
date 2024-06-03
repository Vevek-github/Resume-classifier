# extract.py
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import win32com.client as win32

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_doc(file_path):
    word = win32.Dispatch("Word.Application")
    doc = word.Documents.Open(file_path)
    text = doc.Content.Text
    doc.Close()
    word.Quit()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.doc'):
        return extract_text_from_doc(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return ""

def process_directory(directory, category):
    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f'Processing {file_path}')
            text = extract_text(file_path)
            data.append({"Resume": text, "Category": category})
    return data

def extract_files_to_csv(directory, output_csv_path):
    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f'Processing {file_path}')
            text = extract_text(file_path)
            data.append({"Resume": text})
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data extracted and saved to {output_csv_path}")

# Define directories and categories
directories = {
    r"C:\Users\vevek\OneDrive\Desktop\Python\Resume classifier\Resumes": "React JS Developer",
    r"C:\Users\vevek\OneDrive\Desktop\Python\Resume classifier\Resumes\workday resumes": "Workday",
    r"C:\Users\vevek\OneDrive\Desktop\Python\Resume classifier\Resumes\Peoplesoft resumes": "Peoplesoft",
    r"C:\Users\vevek\OneDrive\Desktop\Python\Resume classifier\Resumes\SQL Developer Lightning insight": "SQL Developer"
}

# Process each directory
all_data = []
for directory, category in directories.items():
    all_data.extend(process_directory(directory, category))

# Create a DataFrame with only two columns: Resume and Category
df = pd.DataFrame(all_data, columns=["Resume", "Category"])
df.to_csv(r"C:\Users\vevek\OneDrive\Desktop\Python\Resume classifier\Resume content\resume.csv", index=False)

print("Data extraction and categorization complete.")
