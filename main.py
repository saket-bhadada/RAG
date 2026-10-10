import json
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"


def main():
    request = Request(
        OLLAMA_URL,
        data=json.dumps(
            {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Say 'setup complete!' in one word"}
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    print(f"Response from Ollama ({MODEL_NAME}): {result['message']['content']}")
    print("Setup complete!")


if __name__ == "__main__":
    main()
