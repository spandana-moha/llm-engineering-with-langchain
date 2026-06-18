import requests

headers = {"User-Agent": "MyLangChainProject/1.0 (student@example.com)"}

url = "https://en.wikipedia.org/w/api.php"

params = {"action": "query", "list": "search", "srsearch": "Artificial Intelligence", "format": "json"}

response = requests.get(url, params=params, headers=headers)

print(response.json())
