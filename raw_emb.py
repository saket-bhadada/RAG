import ollama

# Generates the embedding and extracts the first (and only) vector
vector = ollama.embed(model="nomic-embed-text", input="Say 'yaahooo' in one word")["embeddings"][0]

print(vector)