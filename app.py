import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import testing_model 

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')






st.title("Resume Classifier")
files_list= st.file_uploader("Upload your files in .DOC , .DOCX , .PDF format",accept_multiple_files=True)
predict_button = st.button("Classify the Resume")
if predict_button: 
    df=testing_model.resume_classifier(files_list)
    st.write(df[["Resumes","Prediction"]])
    st.write(df)

        
        
