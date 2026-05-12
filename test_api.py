import requests
import sys

BASE_URL = "http://127.0.0.1:5000"
USER_ID = "emp_001"
IMAGE_PATH = "sample_face.jpg"

def test_health_check():
    print("1. Testing Health Check (/)")
    try:
        response = requests.get(f"{BASE_URL}/")
        print("Status:", response.status_code)
        print("Response:", response.json(), "\n")
    except Exception as e:
        print(f"Error: {e}\n")

def test_register_face():
    print("2. Testing Face Registration (/api/register)")
    try:
        with open(IMAGE_PATH, 'rb') as img:
            files = {'image': img}
            data = {'userid': USER_ID}
            response = requests.post(f"{BASE_URL}/api/register", files=files, data=data)
        print("Status:", response.status_code)
        print("Response:", response.json(), "\n")
    except FileNotFoundError:
        print(f"Error: Could not find image file '{IMAGE_PATH}'. Please ensure the file exists.\n")
    except Exception as e:
        print(f"Error: {e}\n")

def test_check_enrollment():
    print("3. Testing Enrollment Check (/api/is_enrolled)")
    try:
        response = requests.get(f"{BASE_URL}/api/is_enrolled", params={"employeeid": USER_ID})
        print("Status:", response.status_code)
        print("Response:", response.json(), "\n")
    except Exception as e:
        print(f"Error: {e}\n")

def test_match_face():
    print("4. Testing Face Match (/api/match)")
    try:
        with open(IMAGE_PATH, 'rb') as img:
            files = {'image': img}
            data = {'userid': USER_ID}
            response = requests.post(f"{BASE_URL}/api/match", files=files, data=data)
        print("Status:", response.status_code)
        print("Response:", response.json(), "\n")
    except FileNotFoundError:
        print(f"Error: Could not find image file '{IMAGE_PATH}'.\n")
    except Exception as e:
        print(f"Error: {e}\n")

if __name__ == "__main__":
    print("--- Starting API Tests ---\n")
    test_health_check()
    test_register_face()
    test_check_enrollment()
    test_match_face()
    print("--- API Tests Completed ---")
