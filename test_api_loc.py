import requests
import json

# Define the URL for the API endpoint
url = 'https://nutshell-api.azurewebsites.net/process_data'

# Define the data to be sent in the POST request
data = {
    "prediction": "Grievous Injury",
    "len" : 50
}

# Convert the data to JSON format
json_data = json.dumps(data)

# Send the POST request to the API endpoint
response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json_data)

# Print the response from the API
print('Status Code:', response.status_code)
print('Response JSON:', response.json())
print('Response length:', len(list(response.json())))
