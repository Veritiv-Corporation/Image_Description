import requests

url = "http://127.0.0.1:5000/predict"
data = {"key": "value"}  # Replace with your JSON payload
headers = {"Content-Type": "application/json"}  # Set Content-Type header

response = requests.post(url, json=data, headers=headers)
print(response.json())