import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from spire.doc import Document as doct
from spire.doc.common import *


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_doc(file_path):
    # Create a Document object
    document = doct()
    # Load the.doc file
    document.LoadFromFile(file_path)
    # Extract the text of the document
    document_text = document.GetText()
    # Close the document
    document.Close()
    return document_text

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
    #print(f"Data extracted and saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    extract_files_to_csv('path/to/your/doc/files', 'output.csv')
