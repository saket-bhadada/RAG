from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

# Import the Gemini model from LangChain's Google GenAI package
from langchain_google_genai import ChatGoogleGenerativeAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    # Test Gemini
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", # You can also use gemini-1.5-pro 
        temperature=0,
        timeout=None,     # Satisfies the timeout requirement
        max_retries=2
    )
    
    response_gemini = llm_gemini.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatGoogleGenerativeAI: {response_gemini.content}")

    print("Setup complete!")

if __name__ == "__main__":
    main()