# Importing necessary libraries
import pickle
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
from datetime import datetime



# Loading the model, encoder and scaler
model = pickle.load(open("stacking.pkl", "rb"))
binary_encoder = pickle.load(open("binary_encoder.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Creating function to preprocess input and make prediction
def preprocess_and_predict(data):
    reference_date = pd.to_datetime("2010-01-01", format = "%Y-%m-%d")
    data["date"] = pd.to_datetime(data["date"], format = "%Y/%m/%d")
    # Creating the reference date column
    data["days_since_2010"] = (data["date"] - reference_date).dt.days
    # Dropping the date column 
    data.drop("date", axis = 1, inplace = True)
    # Extracting age of the house
    data["house_age"] = 2024 - data["yr_built"]
    # Extracting new category to state whether or not a house has been renovated
    data["house_renovated"] = data["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
    # Dropping the yr_built column and yr_renovated column
    data.drop(["yr_built","yr_renovated"], axis = 1, inplace = True)
    # Converting long and lat from string to float
    data["lat"] = data["lat"].astype("float")  
    data["long"] = data["long"].astype("float")

    # Function to calculate distance from centre in km
    def calculate_distance(row, target_location):
        return geodesic((row["lat"], row["long"]), target_location).km
    # Applying the function to the data
    data["distance_from_centre"] = data.apply(calculate_distance, axis = 1, target_location = (47.56005251931708, -122.21389640494147))
    # Dropping the longitudes and latitudes columns
    data.drop(["lat", "long"], axis = 1, inplace = True)

    # Dropping waterfront and house_renovated 
    data.drop(["waterfront", "house_renovated"], axis = True, inplace = True)

    # Encoding zipcode column 
    data_encoded = binary_encoder.transform(data)
    # Scaling the data 
    data_scaled  = scaler.transform(data_encoded)

    # Making prediction
    prediction = model.predict(data_scaled)[0]

    return prediction

# Building the streamlit app
st.title("House Prices Prediction Model")
st.subheader("Kindly Enter the Features of the House Below 👇👇👇👇👇")

# Taking user input
col1, col2 = st.columns(2)
    
with col1:
    date = st.date_input("Date (YYYY/MM/DD)", min_value=datetime(2010, 1, 1), value=datetime.today())
    sqft_living = st.number_input("Square Feet Living", value=500)
    sqft_lot = st.number_input("Square Feet Lot", value=500)
    sqft_above = st.number_input("Square Feet Above", value=500.00, format="%.2f")
    sqft_basement = st.number_input("Square Feet Basement", value=500)
    zipcode = st.text_input("Zipcode", "98103")
    lat = st.text_input("Latitude", "47.5112")
    long = st.text_input("Longitude", "-122.257")
    sqft_living15 = st.number_input("Square Feet Living (15 nearest neighbors)", value=500)
    sqft_lot15 = st.number_input("Square Feet Lot (15 nearest neighbors)", value=500)
    
with col2:
    waterfront = st.selectbox("Waterfront (Yes: 1, No: 0)", (0, 1))
    bedrooms = st.slider("Bedrooms", min_value=0, max_value=50, value=1, step=1)
    bathrooms = st.slider("Bathrooms", min_value=0.00, max_value=10.00, value=2.00, step=0.25, format="%.2f")
    floors = st.slider("Floors", min_value = 1.0, max_value = 4.0, value = 1.0, step = 0.5, format="%.1f")
    view = st.slider("View", min_value = 0, max_value = 4, value = 0)
    condition = st.slider("Condition", min_value = 1, max_value = 5, value =1)
    grade = st.slider("Grade", min_value = 1, max_value = 13, value = 7)
    yr_built = st.slider("Year Built", min_value=1900, max_value = 2024, value = 1960)
    yr_renovated = st.number_input("Year Renovated (0 if not renovated)", min_value= 0, max_value=2024, value=1960)

# Converting values to dataframe
input_data = pd.DataFrame({
    "date": [date],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft_living": [sqft_living],
    "sqft_lot": [sqft_lot],
    "floors": [floors],
    "waterfront": [waterfront],
    "view": [view],
    "condition": [condition],
    "grade": [grade],
    "sqft_above": [sqft_above],
    "sqft_basement": [sqft_basement],
    "yr_built": [yr_built],
    "yr_renovated": [yr_renovated],
    "zipcode": [zipcode],
    "lat": [lat],
    "long": [long],
    "sqft_living15": [sqft_living15],
    "sqft_lot15": [sqft_lot15]
    })
    
# Prediction
if st.button("Predict"):
    prediction = preprocess_and_predict(input_data)
    st.subheader("Prediction")
    st.write(f"The predicted house price is: ${prediction:,.2f}")
