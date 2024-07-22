from ollama import Client
client = Client(host='http://10.0.0.30:11434')
response = client.chat(model='qwen2-7b-instruct-q8_0:latest', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response)