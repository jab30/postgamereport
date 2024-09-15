import streamlit as st
import pandas as pd
st.set_page_config(page_title='CSV File Reader', layout='wide')
st.header('Single File Upload')
upload_file = st.file_uploader('Upload your CSV File')
df = pd.read_csv(upload_file)
st.dataframe(df, width=1000, height=1200)
