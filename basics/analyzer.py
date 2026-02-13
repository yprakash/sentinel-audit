import os
import time

from dotenv import load_dotenv, find_dotenv
from groq import Groq

dotenv_path = find_dotenv()
loaded = load_dotenv(dotenv_path)
dotenv_path = os.path.relpath(dotenv_path)
print(f"Loaded dotenv from {dotenv_path}: {loaded}")

client = Groq()


def analyze_contract(file_path):
    with open(file_path) as f:
        contract_code = f.read()

    tt = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "You are a Smart Contract Security Strategist. Define the logical invariants for this code."},
            {"role": "user", "content": contract_code}
        ],
        model="llama-3.3-70b-versatile"
    )
    tt = time.time() - tt
    print(f"Time taken for Groq to Analyze contract {file_path}: {tt} seconds")
    return chat_completion.choices[0].message.content


print(analyze_contract("contracts/CoinFlip.sol"))
