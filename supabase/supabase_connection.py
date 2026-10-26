import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

# Note: langchain-postgres works best with the 'postgresql+psycopg://' driver prefix
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/postgres"
)


def connect_to_supabase():
    """Connect to Supabase or local pgvector."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    return vectorstore


def verify_connection(vectorstore):
    """Verify connection by adding, searching, and deleting a test document."""
    test_doc = Document(
        page_content="this is a test document",
        metadata={"test": True},
    )

    try:
        # 1. Pass a list of Document objects: [test_doc]
        ids = vectorstore.add_documents([test_doc])
        print("Added test documents")

        # 2. Run similarity search
        results = vectorstore.similarity_search("test document", k=1)
        if results:
            print(f"Search works: '{results[0].page_content}'")

        # 3. Clean up by passing the returned list of IDs
        vectorstore.delete(ids=ids)
        print("Cleaned up test documents")
        return True
    except Exception as e:
        print(f"Error during verification: {e}")
        return False


def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        print("\n🔗 Connecting to Supabase...")
        host = SUPABASE_URL.split("@")[1].split(":")[0] if "@" in SUPABASE_URL else "configured"
        print(f"   Host: {host}")
    else:
        print("\n⚠️  SUPABASE_DATABASE_URL not set")
        print("   Using local PostgreSQL instead")
        print("\n   To use Supabase:")
        print("   1. Create project at supabase.com")
        print("   2. Go to Settings → Database")
        print("   3. Copy connection string")
        print("   4. Set SUPABASE_DATABASE_URL in .env")

    try:
        vectorstore = connect_to_supabase()
        print("\n✅ Initialized PGVector client!")
    except Exception as e:
        print(f"\n❌ Failed to initialize client: {e}")
        return

    print("\n🧪 Running verification...")
    success = verify_connection(vectorstore)

    if success:
        print("\n✅ Supabase pgvector is ready for production!")
    else:
        print("\n❌ Verification failed - check your connection and Ollama service")


if __name__ == "__main__":
    main()