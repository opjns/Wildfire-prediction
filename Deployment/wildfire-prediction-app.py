import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Define model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 111 * 111, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 111 * 111)
        x = self.fc1(x)
        return x

# Load model weights
model = SimpleCNN()
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Wildfire Prediction App")

st.header("Enter Environmental Data")
lat = st.number_input("Latitude")
lon = st.number_input("Longitude")
ndvi = st.number_input("NDVI")
temp = st.number_input("Temperature (°C)", -20.0, 60.0)
precipitation = st.number_input("Precipitation (mm)")
humidity = st.number_input("Humidity (%)", 0, 100)
windspeed = st.number_input("Wind Speed (m/s)")
day = st.number_input("Day", 1, 31, step=1)
month = st.number_input("Month", 1, 12, step=1)
year = st.number_input("Year")

st.header("Upload Satellite Image")
image_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if image_file and st.button("Predict Wildfire Risk"):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((222, 222)) 
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float()  

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        prediction = int(np.argmax(probabilities))
        confidence = probabilities[prediction]

if prediction == 1:
  st.markdown(f"### Prediction: **{' Wildfire Risk' if prediction == 1 else 'No Wildfire'}**")
  st.markdown(f"### Confidence Level: **{confidence*100:.2f}%**")

if prediction == 1:
  st.warning(" **Safety Precautions**:\n- Avoid open flames\n- Report smoke or fire immediately\n- Keep emergency supplies ready\n- Stay updated with local alerts\n- Prepare evacuation plans")
