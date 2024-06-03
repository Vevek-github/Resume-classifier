# extract_content.py
import os
import pythoncom
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import win32com.client 
import win32com.client

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_doc(file_path):
    # Initialize COM
    pythoncom.CoInitialize()
    try:
        word = win32com.client.Dispatch("Word.Application")
        doc = word.Documents.Open(file_path)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text
    finally:
        # Uninitialize COM
        pythoncom.CoUninitialize()

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

def extract_files_to_csv(directory, output_csv_path):
    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f'Processing {file_path}')
            text = extract_text(file_path)
            data.append({"Resumes": text})
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data extracted and saved to {output_csv_path}")
