import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_predictor_api = Flask("Superkart Sales Predictor")
# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_sales_predictor_api.route("/")
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Predictor API!"

# Define an endpoint for single sales prediction (POST request)
@superkart_sales_predictor_api.route("/v1/predict", methods=["POST"])
def predict():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted product-store sales as a JSON response.
    """
    # Get the JSON data from the request body
    sales_data = request.get_json()
    # Extract relevant features from the JSON data
    sample = {
        "Product_Weight" : sales_data["Product_Weight"],
        "Product_Sugar_Content" : sales_data["Product_Sugar_Content"],
        "Product_Allocated_Area" : sales_data["Product_Allocated_Area"],
        "Product_Type" : sales_data["Product_Type"],
        "Product_MRP" : sales_data["Product_MRP"],
        "Store_Establishment_Year" : sales_data["Store_Establishment_Year"],
        "Store_Size" : sales_data["Store_Size"],
        "Store_Location_City_Type" : sales_data["Store_Location_City_Type"],
       "Store_Type" : sales_data["Store_Type"]
       }
    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame(sample, index=[0])
    # Make prediction
    predicted_sales = model.predict(input_data)

    # Return the actual price
    return jsonify({"predicted_sales": float(predicted_sales[0])})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_predictor_api.route("/v1/predictbatch", methods=["POST"])
def predictbatch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details and returns
    the predicted product-store sales as a JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']
    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)
    feature_columns = ['Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area','Product_Type',
       'Product_MRP', 'Store_Establishment_Year', 'Store_Size',
       'Store_Location_City_Type', 'Store_Type']
    input_data = input_data[feature_columns]

    #Make Prediction
    predicted_sales = model.predict(input_data)


    # Return the predictions dictionary as a JSON response
    return jsonify({"predicted_sales": predicted_sales.tolist()})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_predictor_api.run(debug=True)
