from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from quran_agent.config import OPENAI_API_KEY


def main():
    # Load persisted Chroma vector store
    print("🔄 Loading persisted vector store…")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )
    vectordb = Chroma(
        persist_directory="vectordbs/quran_chroma_db",
        embedding_function=embeddings,
        collection_name="quran_en",
    )
    print("✅ Vector store loaded.")

    # Explore the vector store
    all_docs = vectordb._collection.get()
    if all_docs:
        print("🔍 Exploring vector store:")
        print(f"Number of documents: {len(all_docs['ids'])}")
        print(f"Keys: {all_docs.keys()}")
        print(f"First document ID: {all_docs['ids'][0]}")
        print(f"First document embeddings: {all_docs['embeddings']}")
        print(f"First document content: {all_docs['documents'][0]}")
        print(f"First document metadata: {all_docs['metadatas'][0]}")
        print(f"First document uris: {all_docs['uris']}")
        print(f"First document included: {all_docs['included']}")
        print(f"First document data: {all_docs['data']}")
    else:
        print("No documents found in vectordb.")

    # Build retriever + QA chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm       = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    qa_chain  = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="map_rerank",
        retriever=retriever,
        return_source_documents=True,
    )
    
    # Interactive REPL
    print("\nAsk me anything about the Qur’an (type 'exit' to quit)\n")
    while True:
        query = input("> ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            resp = qa_chain({"question": query})
            print("\n--- Answer ---")
            print(resp["answer"])
            print("\n--- Sources (Surah:Ayah) ---")
            for doc in resp["source_documents"]:
                c = doc.page_content
                md = doc.metadata
                print(f"• {md['surah_name']} ({md['surah_id']}:{md['ayah']}): {c}")
        except ValueError as e:
            print("Model response (unparsed):", e.args[0])

if __name__ == "__main__":
    main()