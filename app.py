import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import testing_model 



with st.sidebar:
    st.image(r"C:\Users\vevek\OneDrive\Desktop\Python\Tele-com churn\social-network-connection-avatar-icon-vector.jpg")
    st.title("Resume Classifier")
    st.info("This is an awesome web app , helps to segment the Resumes .")




st.title("Resume Classifier")
folder_path= st.text_input("Folder Directory")
predict_button = st.button("Predict the Churn")
if predict_button: 
    df=testing_model.resume_classifier(folder_path)
    st.write(df)

        
        
