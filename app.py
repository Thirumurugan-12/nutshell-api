from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from flask_cors import CORS
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('rta_model_deploy3.joblib')

# Load the one-hot encoder
encoder = joblib.load('ordinal_encoder2.joblib')

@app.route('/')
def home():
    return 'RTA Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    Accident_Classification = data['Accident_Classification']
    Accident_Spot = data['Accident_Spot']
    Accident_Location = data['Accident_Location']
    Noofvehicle_involved = data['Noofvehicle_involved']

    # Convert the inputs to a numpy array
    input_array = np.array([[Accident_Classification, Accident_Spot, Accident_Location]], ndmin=2)

    # Encode the input array
    encoded_arr = list(encoder.transform(input_array).ravel())

    # Create the prediction array
    num_arr = [Noofvehicle_involved]
    pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)

    # Make the prediction
    prediction = model.predict(pred_arr)

    # Convert the prediction to a string
    ans = ['Damage Only', 'Fatal', 'Grievous Injury', 'Simple Injury', 'Unknown']
    prediction = ans[prediction[0]]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

@app.route('/process_data', methods=['POST'])
def process_data():
    # Get the input from the request
    data = request.get_json()
    prediction = data['prediction']
    nv = data['len']
    print(prediction)
    print(len)
    #ans = data.get('ans')
    csv_path = data.get('csv_path', 'Copy of Copy of AccidentReports1.csv')

    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df[['Latitude', 'Longitude', 'Severity', 'Road_Condition', 'Weather', 'Main_Cause']].iloc[:60000]
    
    # Filter the data
    df = df[df['Severity'] == prediction][:nv]
    df = df[df['Latitude'] != ""]
    df = df[df['Longitude'] != ""]
    df = df[df['Road_Condition'] != ""]
    
    print(prediction[0])
    #print(ans)
    print(df['Severity'].unique())

    # Rename columns
    df = df.rename(columns={'Latitude'.strip(): 'LATITUDE'.strip(), 'Longitude'.strip(): 'LONGITUDE'.strip()})

    # Convert the DataFrame to a JSON response
    result = df.to_json(orient='records')
    
    return json.loads(result)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)
