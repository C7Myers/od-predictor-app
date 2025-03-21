import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import joblib
from preprocessing import preprocess_image

st.title("📸 Easy OD Predictor App")

os.makedirs('uploaded_images', exist_ok=True)
csv_file = 'od_labels.csv'

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['image_filename', 'od'])
    df.to_csv(csv_file, index=False)

# Load existing data
df = pd.read_csv(csv_file)

# Train model if enough data
if len(df) >= 5:
    X = np.array([preprocess_image(f"uploaded_images/{path}") for path in df['image_filename']])
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

    # Temp save for prediction
    temp_image_path = 'temp.jpg'
    image.save(temp_image_path)

    # Predict OD if model ready
    if prediction_ready:
        features = preprocess_image(temp_image_path).reshape(1, -1)
        predicted_od = model.predict(features)[0]
        st.success(f"🔮 Predicted OD: {predicted_od:.3f}")
    else:
        st.warning("⚠️ Prediction unavailable, more data needed.")

    # Enter Actual OD
    od_value = st.text_input("Enter Actual OD:", "")

    if st.button("Save Image & Actual OD"):
        if od_value:
            try:
                od_float = float(od_value)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                image_filename = f'image_{timestamp}.jpg'
                os.rename(temp_image_path, f'uploaded_images/{image_filename}')

                # Save entry to CSV
                new_entry = pd.DataFrame({'image_filename': [image_filename], 'od': [od_float]})
                new_entry.to_csv(csv_file, mode='a', header=False, index=False)

                st.success("✅ Data saved! Your model will continuously improve.")

            except ValueError:
                st.error("❌ Enter a valid numeric OD value.")
        else:
            st.error("❌ OD value is required.")

# Easy download button to get your data clearly
if os.path.exists(csv_file):
    with open(csv_file, 'rb') as f:
        st.download_button('📥 Download Data (CSV)', f, file_name='od_labels.csv')
