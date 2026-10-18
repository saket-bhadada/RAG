from langchain_ollama import OllamaEmbeddings, ChatOllama
import numpy as np

# 1. Use a dedicated embedding model for converting text to vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Keep qwen2.5:7b for ChatOllama when you need to generate answers later
llm = ChatOllama(model="qwen2.5:7b")

def basic_embeddings():
    text = "what is ml?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm:  {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():


if __name__ == "__main__":
    basic_embeddings()