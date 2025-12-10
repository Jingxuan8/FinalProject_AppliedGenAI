import openai

from openai import OpenAI

client = OpenAI(api_key=REDACTED" | Out-File replace.txt -Encoding utf8")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Name an animal that starts with the letter P."}
    ]
)

print("Assistant:", resp.choices[0].message.content)