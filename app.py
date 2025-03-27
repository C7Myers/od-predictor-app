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

st.title("📸 OD Predictor App")

# ✅ Define OAuth Scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ✅ Load credentials securely from Streamlit Secrets
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

# ✅ Authenticate Google APIs
client = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)

# ✅ Your Google Drive folder ID
image_subfolder_id = '11IXxbuYT7gAvd4Yggn1GtAd2sv0FF_RO'

# ✅ Open or create Google Sheet for OD data
sheet_name = "OD_App_Data"

sheet = client.open(sheet_name).sheet1
df = pd.DataFrame(sheet.get_all_records())
entry = len(df) + 1


# ✅ Function to download images from Google Drive
def download_image_from_drive(service, file_name, output_path, image_subfolder_id):
    """Download an image from Google Drive given its name and folder."""
    query = f"name='{file_name}' and '{image_subfolder_id}' in parents"
    results = service.files().list(q=query, spaces='drive').execute()
    
    files = results.get('files', [])
    
    if not files:
        st.error(f"❌ Image {file_name} not found in Drive.")
        return None
    
    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    
    with open(output_path, "wb") as f:
        f.write(request.execute())
    
    return output_path  # ✅ Local path to downloaded image

# ✅ Train model
X = []
y = []
    
for _, row in df.iterrows():
    local_path = f"temp_downloaded_{row['image_filename']}"
    downloaded_path = download_image_from_drive(drive_service, row["image_filename"], local_path, image_subfolder_id)
        
    if downloaded_path:  # ✅ Only process if download was successful
        X.append(preprocess_image(downloaded_path))
        y.append(row["od"])

X = np.array(X)
y = np.array(y)

model = RandomForestRegressor(n_estimators=50)
model.fit(X, y)

# ✅ Upload Image
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Image', use_container_width=True)

    temp_image_path = 'temp.jpg'
    image.save(temp_image_path)

    # ✅ Predict OD based on model 
    features = preprocess_image(temp_image_path).reshape(1, -1)
    predicted_od = model.predict(features)[0]
    st.success(f"🔮 Predicted OD: {predicted_od:.3f}")

    od_value = st.text_input("Enter Actual OD:", "")

    if st.button("Save Image & Actual OD"):
        if od_value:
            try:
                od_float = float(od_value)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                image_filename = f'image_{timestamp}.jpg'

                # ✅ Upload image to Google Drive
                file_metadata = {'name': image_filename, 'parents': [image_subfolder_id]}
                media = MediaFileUpload(temp_image_path, mimetype='image/jpeg')
                drive_service.files().create(body=file_metadata, media_body=media).execute()

                # Calculate prediction and deviation
                predicted_od = model.predict(features)[0] if prediction_ready else None
                deviation = abs(od_float - predicted_od) if predicted_od is not None else None

                # ✅ Save OD entry to Google Sheets
                sheet.append_row([entry, image_filename, od_float, predicted_od, deviation])

                os.remove(temp_image_path)

                st.success("✅ Thank you for helping train the model!")

                df1 = pd.DataFrame(sheet.get_all_records())

                # Only plot if deviation column exists and isn't all blank
                if "deviation" in df1.columns and not df1["deviation"].isnull().all():
                    st.subheader("📉 Model Deviation Over Time")
                    st.line_chart(df1["deviation"])

                # ✅ Stats
                df1["deviation"] = pd.to_numeric(df1["deviation"], errors="coerce")
                current_dev = df1["deviation"].iloc[-1]
                avg_dev = df1["deviation"].mean()

                st.write(f"📈 Current Deviation: `{current_dev:.3f}`")
                st.write(f"📊 Average Deviation: `{avg_dev:.3f}`")

                if current_dev < avg_dev:
                    st.success("✅ Model performance is improving!")
                else:
                    st.warning("📉 Prediction is above average deviation.")
            
            except ValueError:
                st.error("❌ Enter a valid numeric OD value.")
        else:
            st.error("❌ OD value is required.")

   


    

