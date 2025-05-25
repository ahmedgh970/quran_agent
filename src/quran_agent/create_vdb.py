from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from quran_agent.quran_client import QuranClient
from quran_agent.config import (
    PROD_CLIENT_ID,
    PROD_BASE_URL,
    PROD_ACCESS_TOKEN,
    OPENAI_API_KEY,
)

def main():
    """
    Main function to create a vector database from Quran documents.
    """
    # Authenticate & build the data client
    qc = QuranClient(
        PROD_BASE_URL,
        PROD_ACCESS_TOKEN,
        PROD_CLIENT_ID,
    )

    # Stream in your Documents
    print("⏳ Fetching and preparing Documents…")
    documents = list(qc.iter_documents())
    print(f"✅ Prepared {len(documents)} documents.")
    print(f"First document content: {documents[0].page_content}")
    print(f"First document metadata: {documents[0].metadata}")

    # Init embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )

    # Build & persist the chroma vector store
    Chroma.from_documents(
        documents,
        embeddings,
        persist_directory="vectordbs/quran_chroma_db",
        collection_name="quran_en",
    )
    print("🎉 Vector store created and persisted at ./vectordbs/quran_chroma_db")

if __name__ == "__main__":
    main()