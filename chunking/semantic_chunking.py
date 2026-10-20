from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# from text_splitter import recursive_splitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")

document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""

#step 1 initial chunking using recursivecharactertextsplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=['\n\n','\n','. ',' ']
)

recursive_chunks = recursive_splitter.split_text(document)

print(f"Recursive chunks: {len(recursive_chunks)}")

for i,chunk in enumerate(recursive_chunks):
    print(f"Recursive chunk {i}: {chunk}")
    print(chunk[:100] + '...' if len(chunk) > 100 else chunk)


#semantic chunking

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_amount=90,
    breakpoint_threshold_type='percentile'
)

semantic_chunks = semantic_chunker.split_text(document)
print(f"Semantic chunks: {len(semantic_chunks)}")
for i,chunk in enumerate(semantic_chunks):
    print(f"\n\nSemantic chunk {i}: {chunk}")
    print(chunk[:100] + '...' if len(chunk) > 100 else chunk)