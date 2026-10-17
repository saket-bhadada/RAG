import tempfile
from re import search
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chat_models import init_chat_model

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)
# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0
# )
# llm = init_chat_model("llama3.1", model_provider="ollama", temperature=0)
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""
#defining knowledge base
def create_kb():
    """create knowledge base"""
    # splitting text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,
                   metadate={"source":"langchain_knowledge_base.md"})
    chunks = splitter.split_documents([doc])

    vector_Store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_Store

def demo_basic_rag():
    vector_Store = create_kb()
    retrieval = vector_Store.as_retriever(search_type="similarity",search_kwargs={"k":2})
    llm = init_chat_model(
        model="qwen2.5:7b",
        model_provider="ollama",
        temperature=0.2,
        streaming=False,
        num_gpu=0,  # Forces Ollama to compute on CPU directly from Python
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:
        {context}
        Question:{question}
        Answer:
        
        make sure to ans in the concise manner, and if you dont know just say "i dont know"."""
    )

    def format_docs(docs:List[Document])->str:
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
            {"context": retrieval | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    question = [
        "what is langchain?",
        "who created langchain?",
        "what is langgraph used for?"
    ]
    print("basic rag demo:\n")
    for q in question:
        answer = rag_chain.invoke(q)
        print(q)
        print(answer)


if __name__ == "__main__":
    demo_basic_rag()