import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime

st.title("📸 OD Image Uploader")

os.makedirs('uploaded_images', exist_ok=True)

csv_file = 'od_labels.csv'
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['image_filename', 'od'])
    df.to_csv(csv_file, index=False)

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    od_value = st.text_input("Enter Actual OD (required):", "")

    if st.button("Save Image and OD"):
        if od_value:
            try:
                od_float = float(od_value)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                image_filename = f'image_{timestamp}.jpg'
                image.save(f'uploaded_images/{image_filename}')

                new_entry = pd.DataFrame({'image_filename': [image_filename], 'od': [od_float]})
                new_entry.to_csv(csv_file, mode='a', header=False, index=False)

                st.success("✅ Image and OD saved successfully!")

            except ValueError:
                st.error("❌ Please enter a valid numeric OD value.")
        else:
            st.error("❌ OD value is required to save.")
