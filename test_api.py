import requests
import json

# Set the API endpoint URL
url = 'http://localhost:5000/predict'

# Define the input data
input_data = {
    'Noofvehicle_involved': 2,
    'Accident_Classification': 'Road Accidents',
    'Accident_Spot': 'Curves',
    'Accident_Location': 'Rural Areas'
}

# Convert the input data to JSON
json_data = json.dumps(input_data)

# Set the headers
headers = {'Content-Type': 'application/json'}

# Send the POST request to the API
response = requests.post(url, data=json_data, headers=headers)

# Check the response status code
if response.status_code == 200:
    # Get the prediction from the response JSON
    prediction = response.json()['prediction']
    print(f'Prediction: {prediction}')
else:
    print(f'Error: {response.status_code} - {response.text}')