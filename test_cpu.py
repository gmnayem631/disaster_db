from ollama import Client
from pydantic import BaseModel

class Tiny(BaseModel):
    label: str
    value: int

client = Client(host="http://127.0.0.1:11434", timeout=600)
opts = {"temperature": 0, "num_gpu": 0, "num_predict": 128}

def run(name, prompt, fmt=None):
    r = client.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        think=False, stream=False, keep_alive="10m",
        format=fmt, options=opts,
    )
    print(f"{name}: content={r.message.content!r} "
          f"done={r.done_reason} eval={r.eval_count}")

for i in range(3):
    run(f"plain {i+1}", "Reply with the single word OK.")
for i in range(3):
    run(f"json  {i+1}", "Answer with label=FATALITIES and value=31, as JSON.",
        Tiny.model_json_schema())