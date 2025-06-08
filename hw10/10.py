import ollama

response = ollama.chat(
    model="llama3:8b",  
    messages=[
        {"role": "user", "content": "1+1=?"}
    ]
)


print(response["message"]["content"])

