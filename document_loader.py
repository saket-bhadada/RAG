import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import (TextLoader)

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False,suffix=".txt") as temp_file:
        temp_file.write(b"This is temp file content")
        temp_file_path = temp_file.name
    
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for docs in documents:
            print("content in the document:")
            print(docs)
            print(docs.page_content)
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    load_text_file()