import ollama

response = ollama.chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "What is a resume?"
        }
    ]
)

print(response["message"]["content"])