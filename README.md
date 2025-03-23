# 📸 OD Predictor App

A web app for predicting optical density (OD) of liquid cultures using uploaded images. Built with **Streamlit**, powered by a **Random Forest** model that continuously improves as more data is added.

📍 **Live app**: [od-predictor-app](https://od-predictor-app-yejdzjz2gcdbc9ynfd7kya.streamlit.app/)

## 🚀 How it Works

1. Upload an image of your culture (e.g. test tube with Kimwipe background).
2. Enter the actual OD value (optional but improves training).
3. The app predicts the OD using a trained machine learning model.
4. Your image is saved to Google Drive, and OD data is stored in Google Sheets.
5. After 5+ samples, the model automatically retrains to improve accuracy.

## 📦 Tech Stack

- **Streamlit**: Web app interface
- **Google Drive API**: Store uploaded images
- **Google Sheets API**: Log OD data and predictions
- **scikit-learn**: Random Forest model
- **Pillow / NumPy / pandas**: Image preprocessing and data handling

## 🔄 Continuous Learning

The model is retrained every time the app runs, using all available data:
- Input features are extracted from each image.
- OD values from Google Sheets serve as training labels.
- Predicted OD and deviation are also logged for tracking performance.

## 📁 File Structure

- `app.py`: Main Streamlit app
- `preprocessing.py`: Image preprocessing logic
- `README.md`: This file
- Google Drive: Stores uploaded images in an `Images/` subfolder
- Google Sheets: Stores all OD logs (`OD_App_Data`)

## 🔐 Notes

- Authentication is handled via a service account.
- API credentials are securely managed via Streamlit secrets.
---

Made by Callum Myers 
