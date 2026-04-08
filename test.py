import time
import requests

# Change this to your endpoint
url = "http://localhost:5001/v1/chat/completions" 
headers = {"Content-Type": "application/json"}
data = {
    "model": "model-name-here",
    "messages": [{"role": "user", "content": "Write a 500 word story about a cat."}],
    "max_tokens": 500,
    "stream": False # Set to True if you want to measure streaming speed
}

start_time = time.time()
response = requests.post(url, headers=headers, json=data)
end_time = time.time()

if response.status_code == 200:
    res_json = response.json()
    tokens = res_json['usage']['completion_tokens']
    duration = end_time - start_time
    tps = tokens / duration
    print(f"Tokens: {tokens}")
    print(f"Time: {duration:.2f}s")
    print(f"TPS: {tps:.2f} tokens/sec")
else:
    print(f"Error: {response.status_code}")