import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 32 * 32, 1)  # Assuming binary output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc1(x)
        return x

# Load the model weights
model = SimpleCNN()
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Wildfire Prediction")

st.header("Enter Environmental Data")
lat = st.number_input("Latitude")
lon = st.number_input("Longitude")
ndvi = st.number_input("NDVI")
temp = st.number_input("Temperature (°C)", -20.0, 60.0)
humidity = st.slider("Humidity (%)", 0, 100)
day = st.number_input("Day", 1, 31, step=1)
month = st.number_input("Month", 1, 12, step=1)
year = st.number_input("Year")

# Image
st.header("Upload Satellite Image")
image_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
image_features = []

if image_file:
    image = Image.open(image_file).convert("RGB")
    image = image.resize((64, 64))
    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

    image_array = np.array(image) / 255.0
    image_features = image_array.flatten()

# Predict button
if st.button("Predict Wildfire Risk"):
    tabular_features = [lat, lon, ndvi, temp, humidity, day, month, year]
    final_input = np.array(tabular_features + list(image_features)).reshape(1, -1)

    prediction = model.predict(final_input)[0]
    confidence = model.predict_proba(final_input)[0][1]  # Probability of class 1 (fire)

    st.markdown(f"### Prediction: **{'Risk of Wildfire' if prediction == 1 else 'No Wildfire'}**")
    st.markdown(f"### Confidence Level: **{confidence*100:.2f}%**")

    if prediction == 1:
        st.warning("🚨 **Safety Precautions**:\n- Avoid open flames\n- Report smoke or fire immediately\n- Keep emergency supplies ready\n- Stay updated with local alerts\n- Prepare evacuation plans")
