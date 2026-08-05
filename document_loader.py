import os
import tempfile
import warnings

from dotenv import load_dotenv

# Suppress the DeprecationWarning triggered by langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Now we can import TextLoader without the terminal yelling at us
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from pathlib import Path

load_dotenv()


def load_text_file():
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is temp file content")
        temp_file_path = temp_file.name

    try:
        # Use TextLoader as requested
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # for docs in documents:
        #     print("content in the document:")
        #     print(docs)
        #     print(docs.page_content)
        print(f"loaded {len(documents)} documents")
        print(f"content preview {documents[0].page_content[:100]}...")
        print(f"metadata {documents[0].metadata}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def pdf_loader(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"loaded {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"document {i+1}")
        print(f"content preview {doc.page_content[:100]}...")
        print(f"metadata {doc.metadata}")


if __name__ == "__main__":
    # load_text_file()
    pdf_loader("Digital_Heroes_Full_Stack_Developer_Trial.pdf")