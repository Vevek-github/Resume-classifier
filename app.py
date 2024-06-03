import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import testing_model 







st.title("Resume Classifier")
folder_path= st.text_input("Folder Directory")
predict_button = st.button("Predict the Churn")
if predict_button: 
    df=testing_model.resume_classifier(folder_path)
    st.write(df)

        
        
