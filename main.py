<<<<<<< HEAD
import streamlit as st
import pandas as pd
from PIL import Image
import os
from fastai.vision.all import load_learner
import pickle
import pathlib
import torch
from torchvision import transforms
from fastai.data.transforms import Normalize

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


rainpred = pickle.load(open(r'rain\rainfall_prediction_model.pkl', 'rb'))

feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine',
       'winddirection', 'windspeed']

wheat = load_learner(r'C:\Projects\mini-project-sem3\Models\wheat.pkl', pickle_module=pickle)
rice = load_learner(r'C:\Projects\mini-project-sem3\Models\rice.pkl', pickle_module=pickle)
sugarcane = load_learner(r'C:\Projects\mini-project-sem3\Models\sugarcane.pkl', pickle_module=pickle)
tomato = load_learner(r'C:\Projects\mini-project-sem3\Models\tomato.pkl', pickle_module=pickle)

disease_models = {
    'wheat': wheat,
    'rice': rice,
    'sugarcane': sugarcane,
    'tomato': tomato
}

def predict_crop_disease(crop, image):
    model = disease_models[crop]
    prediction, idx, probs = model.predict(image)
    return prediction, idx, probs



def predict_rainfall(input_df):
    model = rainpred['model']
    expected_features = rainpred['feature_names']
    input_df = input_df[expected_features] 
    rainfall = model.predict(input_df)
    
    if rainfall[0] == 1:
        return "It may rain today."
    else:
        return "It may not rain today."


def preprocess_image_for_fastai(image, model_size=224):
    from fastai.vision.all import PILImage
    fastai_image = PILImage.create(image)
    fastai_image = fastai_image.resize((model_size, model_size))
    return fastai_image

# def predict_crop_price(crop, date):
#     # Use your crop price forecasting model
#     price = price_model.predict(crop, date)
#     return price

st.set_page_config(page_title="FarmWise")

st.sidebar.title("FarmWise")
app_mode = st.sidebar.selectbox("Select task", ["Predict Crop Disease", "Predict Rainfall", "Predict Crop Price"])

if app_mode == "Predict Crop Disease":
    st.image("C:\Projects\mini-project-sem3\images\Crop Disease Detection.png", caption = 'Upload image -> Select crop name -> Predict Disease' , use_container_width=True)
    # st.title("Crop Disease Detection")
    crop = st.selectbox("Select Crop", list(disease_models.keys()))


    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        preprocessed_image = preprocess_image_for_fastai(image)

        if st.button("Predict"):
            prediction, idx, probs = predict_crop_disease(crop=crop, image=preprocessed_image)
            st.write(f"Prediction: {prediction}")
            st.write(f"Confidence: {probs[idx]*100:.2f}%")
            
elif app_mode == "Predict Rainfall":
    st.image("C:\Projects\mini-project-sem3\images\Rain.png", caption = 'Input values -> Predict chances of rainfall.' , use_container_width=True)
    pressurex = st.number_input("Pressure: ", value=1015.9)
    dewpointx = st.number_input("Dewpoint: ", value=19.9)
    humidityx = st.number_input("Humidity: ", value=95)
    cloudx = st.number_input("Cloud: ", value=80)
    sunshinex = st.radio("Sunshine: ", options=["Yes","No"])
    sunshinex = 0.0 if sunshinex == "No" else 1.0
    windx = st.number_input("Wind Direction: ", value=40)
    windspeedx = st.number_input("Wind Speed: ", value=13.7)
    
    ipdf = (pressurex, dewpointx, humidityx, cloudx, sunshinex, windx, windspeedx)
    ipdf = pd.DataFrame([ipdf], columns=feature_names)
    if st.button("Predict"):
        rainfall = predict_rainfall(input_df=ipdf)
        st.write(f"Prediction: {rainfall}")
        
# elif app_mode == "Predict Crop Price":
#     st.title("Crop Price Forecasting")
#     crop = st.selectbox("Select Crop", ["wheat", "rice", "sugarcane", "tomato", "chickpea"])
#     date = st.date_input("Select Date")
    
#     if st.button("Predict"):
#         price = predict_crop_price(crop, date)
=======
import streamlit as st
import pandas as pd
from PIL import Image
import os
from fastai.vision.all import load_learner
import pickle
import pathlib
import torch
from torchvision import transforms
from fastai.data.transforms import Normalize

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


rainpred = pickle.load(open(r'rain\rainfall_prediction_model.pkl', 'rb'))

feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine',
       'winddirection', 'windspeed']

wheat = load_learner(r'C:\Projects\mini-project-sem3\Models\wheat.pkl', pickle_module=pickle)
rice = load_learner(r'C:\Projects\mini-project-sem3\Models\rice.pkl', pickle_module=pickle)
sugarcane = load_learner(r'C:\Projects\mini-project-sem3\Models\sugarcane.pkl', pickle_module=pickle)
tomato = load_learner(r'C:\Projects\mini-project-sem3\Models\tomato.pkl', pickle_module=pickle)

disease_models = {
    'wheat': wheat,
    'rice': rice,
    'sugarcane': sugarcane,
    'tomato': tomato
}

def predict_crop_disease(crop, image):
    model = disease_models[crop]
    prediction, idx, probs = model.predict(image)
    return prediction, idx, probs



def predict_rainfall(input_df):
    model = rainpred['model']
    expected_features = rainpred['feature_names']
    input_df = input_df[expected_features] 
    rainfall = model.predict(input_df)
    
    if rainfall[0] == 1:
        return "It may rain today."
    else:
        return "It may not rain today."


def preprocess_image_for_fastai(image, model_size=224):
    from fastai.vision.all import PILImage
    fastai_image = PILImage.create(image)
    fastai_image = fastai_image.resize((model_size, model_size))
    return fastai_image

# def predict_crop_price(crop, date):
#     # Use your crop price forecasting model
#     price = price_model.predict(crop, date)
#     return price

st.set_page_config(page_title="FarmWise")

st.sidebar.title("FarmWise")
app_mode = st.sidebar.selectbox("Select task", ["Predict Crop Disease", "Predict Rainfall", "Predict Crop Price"])

if app_mode == "Predict Crop Disease":
    st.image("C:\Projects\mini-project-sem3\images\Crop Disease Detection.png", caption = 'Upload image -> Select crop name -> Predict Disease' , use_container_width=True)
    # st.title("Crop Disease Detection")
    crop = st.selectbox("Select Crop", list(disease_models.keys()))


    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        preprocessed_image = preprocess_image_for_fastai(image)

        if st.button("Predict"):
            prediction, idx, probs = predict_crop_disease(crop=crop, image=preprocessed_image)
            st.write(f"Prediction: {prediction}")
            st.write(f"Confidence: {probs[idx]*100:.2f}%")
            
elif app_mode == "Predict Rainfall":
    st.image("C:\Projects\mini-project-sem3\images\Rain.png", caption = 'Input values -> Predict chances of rainfall.' , use_container_width=True)
    pressurex = st.number_input("Pressure: ", value=1015.9)
    dewpointx = st.number_input("Dewpoint: ", value=19.9)
    humidityx = st.number_input("Humidity: ", value=95)
    cloudx = st.number_input("Cloud: ", value=80)
    sunshinex = st.radio("Sunshine: ", options=["Yes","No"])
    sunshinex = 0.0 if sunshinex == "No" else 1.0
    windx = st.number_input("Wind Direction: ", value=40)
    windspeedx = st.number_input("Wind Speed: ", value=13.7)
    
    ipdf = (pressurex, dewpointx, humidityx, cloudx, sunshinex, windx, windspeedx)
    ipdf = pd.DataFrame([ipdf], columns=feature_names)
    if st.button("Predict"):
        rainfall = predict_rainfall(input_df=ipdf)
        st.write(f"Prediction: {rainfall}")
        
# elif app_mode == "Predict Crop Price":
#     st.title("Crop Price Forecasting")
#     crop = st.selectbox("Select Crop", ["wheat", "rice", "sugarcane", "tomato", "chickpea"])
#     date = st.date_input("Select Date")
    
#     if st.button("Predict"):
#         price = predict_crop_price(crop, date)
>>>>>>> c792be16b1c204ce7c24c6f2c6c0933d9d78d0d0
#         st.write(f"Predicted Price: {price} per unit")