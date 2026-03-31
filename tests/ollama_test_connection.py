import requests

try:
    connection = requests.get("http://localhost:11434")
    print("OK:", connection)
except Exception as e:
    print("ERROR:", e)