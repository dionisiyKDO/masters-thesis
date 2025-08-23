import requests
import json

def register_user(username, email, password, role):
    url = 'http://localhost:8000/api/auth/register/'
    data = {'username': username, 'email': email, 'password': password, 'role': role}
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    return url, json_data, headers

url, json_data, headers = register_user('dionisiy', 'dionisiykd@gmail.com', 'dionisiy', 'admin')
response = requests.post(url, data=json_data, headers=headers)
print('Register user:')
print(f'Status Code: {response.status_code}')
print(f'Response JSON: {response.json()}')

url, json_data, headers = register_user('dionisiy1', 'dionisiykd@gmail.com', 'dionisiy1', 'doctor')
response = requests.post(url, data=json_data, headers=headers)
print('Register user:')
print(f'Status Code: {response.status_code}')
print(f'Response JSON: {response.json()}')

url, json_data, headers = register_user('dionisiy2', 'dionisiykd@gmail.com', 'dionisiy2', 'patient')
response = requests.post(url, data=json_data, headers=headers)
print('Register user:')
print(f'Status Code: {response.status_code}')
print(f'Response JSON: {response.json()}')


def get_token():
    url = 'http://localhost:8000/api/auth/token/'
    data = {'username': 'dionisiy', 'password': 'dionisiy'}
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    return url, json_data, headers

url, json_data, headers = get_token()
response = requests.post(url, data=json_data, headers=headers)

print('Get token:')
print(f'Status Code: {response.status_code}')
print(f'Response JSON: {response.json()}')

def get_users(access_token):
    url = 'http://localhost:8000/api/auth/users/'
    data = {}
    json_data = json.dumps(data)
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    return url, json_data, headers

url, json_data, headers = get_users(access_token=response.json()['access'])
response = requests.get(url, headers=headers)

print('Get all users:')
print(f'Status Code: {response.status_code}')
print(f'Response JSON: {response.json()}')

# def create_comment(access_token):
#     url = 'http://localhost:8000/api/blog/comments/'
#     # data = { 'body': 'Test comment', 'user': 1, 'post': 1 }
#     data = { 'body': 'Test comment', 'post': 1 }
#     json_data = json.dumps(data)
#     headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
#     return url, json_data, headers

# url, json_data, headers = create_comment(access_token=response.json()['access'])
# response = requests.post(url, data=json_data, headers=headers)

# print('Get all users:')
# print(f'Status Code: {response.status_code}')
# print(f'Response JSON: {response.json()}')