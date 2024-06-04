import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from spire.doc import Document as SpireDoc
from spire.doc.common import *
import tempfile


def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_doc(file):
    # Save the uploaded DOC file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    doci = SpireDoc()
    
    # Load the.doc file
    doci.LoadFromFile(temp_file_path)
    # Extract the text of the document
    document_text = doci.GetText()
    # Close the document
    doci.Close()

    return document_text[70:]

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.endswith('.doc'):
        return extract_text_from_doc(file)
    elif file.name.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        return ""

def extract_from_files(files_list):
    data = []
    for file in files_list:
        print(f'Processing {file.name}')
        text = extract_text(file)
        data.append({"Resumes": text})
    df = pd.DataFrame(data)
    #df.to_csv("https://github.com/Vevek-github/Resume-classifier/raw/7b58fc114f07d0a053b13420a5dea836fa94c5df/Resume%20content/test_resume.csv, index=False")
    #print(f"Data extracted and saved to {output_csv_path}")
    return df 

# Example usage
if __name__ == "__main__":
    pass
