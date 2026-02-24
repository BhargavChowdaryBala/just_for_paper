import requests
import json

url = 'http://127.0.0.1:5000/api/analyze'
file_path = 'c:/Users/bharg/Desktop/just_for_paper/test_plate.jpg'

try:
    with open(file_path, 'rb') as f:
        files = {'image': f}
        print(f"Sending {file_path} to {url}...")
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Response Data:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Failed! Error:")
        print(response.text)
except Exception as e:
    print(f"Error during request: {e}")
