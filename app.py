import streamlit as st
from PIL import Image
from src import run_model, process_image, load_model
import os

MODEL_PATH = './model/best.pt'
model = load_model(MODEL_PATH)

st.set_page_config(page_title='YOLOv10 Helmet detection',
                   layout='wide',
                   initial_sidebar_state='auto',
                   page_icon='🍍',
                   menu_items={
                       'About': 'This is a simple app.'
                   }
                   )

st.title('YOLOv10 Helmet detection')
st.write('Upload an image to detect helmets')

uploaded_file = st.file_uploader(
    "Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 2])
    with col1:
        img = Image.open(uploaded_file)
        st.image(uploaded_file, use_column_width=True)

    with col2:
        with st.spinner('Processing...'):
            results = process_image(run_model(img, model))
        st.image(results, caption='Result', use_column_width=True)
