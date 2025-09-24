from openai import OpenAI
import json, os

client = OpenAI(api_key=API_KEY)

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # 権限のあるモデル名を使う
    messages=[
        {"role":"system","content":"Output valid JSON only."},
        {"role":"user","content":"Return {\"ping\":\"pong\"} as strict JSON."},
    ],
    response_format={"type":"json_object"},
    temperature=0.2,
)
print(json.loads(resp.choices[0].message.content))