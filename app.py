import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import joblib
from preprocessing import preprocess_image
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

st.title("📸 Easy OD Predictor App")

# Connect to Google Drive
scopes = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
creds = Credentials.from_service_account_file("your-json-key.json", scopes=scopes)
client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)

# Your Google Drive folder ID clearly stated
folder_id = '1gaU-WUZesT9E4VXRnIs6H4NVslI861tk'

# Open or create Google Sheet clearly
sheet_name = "OD_App_Data"
try:
    sheet = client.open(sheet_name).sheet1
    df = pd.DataFrame(sheet.get_all_records())
except:
    sheet = client.create(sheet_name).sheet1
    sheet.append_row(['image_filename', 'od'])
    df = pd.DataFrame(columns=['image_filename', 'od'])

# Train model clearly
if len(df) >= 5:
    X = np.array([preprocess_image(f"temp_downloaded_{row['image_filename']}") for _, row in df.iterrows()])
    y = df['od'].values
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    prediction_ready = True
else:
    prediction_ready = False
    model = None
    st.info(f"ℹ️ Add {5 - len(df)} more images to start predictions.")

# Upload Image
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Image', use_container_width=True)

    temp_image_path = 'temp.jpg'
    image.save(temp_image_path)

    # Predict OD if model ready
    if prediction_ready:
        features = preprocess_image(temp_image_path).reshape(1, -1)
        predicted_od = model.predict(features)[0]
        st.success(f"🔮 Predicted OD: {predicted_od:.3f}")
    else:
        st.warning("⚠️ Prediction unavailable, more data needed.")

    od_value = st.text_input("Enter Actual OD:", "")

    if st.button("Save Image & Actual OD"):
        if od_value:
            try:
                od_float = float(od_value)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                image_filename = f'image_{timestamp}.jpg'

                # Upload to Google Drive
                file_metadata = {'name': image_filename, 'parents': [folder_id]}
                media = MediaFileUpload(temp_image_path, mimetype='image/jpeg')
                drive_service.files().create(body=file_metadata, media_body=media).execute()

                # Save entry to Google Sheet
                sheet.append_row([image_filename, od_float])

                os.remove(temp_image_path)

                st.success("✅ Data permanently stored in Google Drive!")

            except ValueError:
                st.error("❌ Enter a valid numeric OD value.")
        else:
            st.error("❌ OD value is required.")
