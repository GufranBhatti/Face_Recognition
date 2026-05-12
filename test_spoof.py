import requests

BASE_URL = "http://127.0.0.1:5000"
USER_ID = "emp_spoof"
IMAGE_PATH = "fake_face.jpg"

print("Testing Spoof Registration (/api/register)")
try:
    with open(IMAGE_PATH, 'rb') as img:
        files = {'image': img}
        data = {'userid': USER_ID}
        response = requests.post(f"{BASE_URL}/api/register", files=files, data=data)
    print("Status:", response.status_code)
    print("Response:", response.json(), "\n")
except Exception as e:
    print(f"Error: {e}\n")
