from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics():
    # Omit persist_directory to create an ephemeral, in-memory database
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        collection_name="chroma_basics",
    )
    document_ids = vectorstore.get(include=[]).get("ids", [])
    print(f"Vector store created {len(document_ids)} documents in-memory.")

    # perform similarity search
    query = "What is LangChain?"
    results: list[Document] = vectorstore.similarity_search(query, k=2)

    print(f"\nTop 2 results for query '{query}':")
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        print(f"Result {i + 1}: {doc.page_content} (Source: {source})")


def similarity_search():
    # Omit persist_directory here as well
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        collection_name="similarity_search",
    )

    query = "Explain vector stores"
    result_with_score = vectorstore.similarity_search_with_score(query, k=3)

    print(f"\nTop 3 results with scores for query '{query}':")
    for i, (doc, score) in enumerate(result_with_score):
        final_score = 1 / (1 + score)
        print(f"Document {i + 1}: {doc.page_content} (Score: {final_score:.4f})")


def metadata_filtering():
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        collection_name="metadata_filtering",
    )

    query = "What database are available?"
    results: list[Document] = vectorstore.similarity_search(query, k=5)
    print(f"\nTop 5 results for query '{query}':")
    for i, doc in enumerate(results):
        print(f"result {i + 1}: {doc.page_content}")

    filtering_criteria: dict[str, str] = {"topic": "database"}
    filtering_results: list[Document] = vectorstore.similarity_search(
        query, k=5, filter=filtering_criteria
    )
    print(f"\nTop 5 filtered results for query '{query}':")
    for i, doc in enumerate(filtering_results):
        print(f"result {i + 1}: {doc.page_content}")


metadate_filtering = metadata_filtering


if __name__ == "__main__":
    chroma_basics()
    similarity_search()
    metadata_filtering()