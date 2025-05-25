import re
from typing import Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from quran_agent.config import OPENAI_API_KEY


def explore_vdb(vectordb: Chroma):
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

def load_vectorstore(
    persist_dir: str = "vectordbs/quran_chroma_db"
    ) -> Chroma:
    """
    Load an existing Chroma vector store from disk.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )
    # Depending on your Chroma version, you may need embedding_function=embeddings
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="quran_en",
    )
    return vectordb

def build_prompt() -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """
    Creates a ChatPromptTemplate that instructs the LLM
    to think step-by-step over retrieved Qur’an verses.
    """
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_system_prompt = (
        "You are an expert, knowledgeable Qur’an scholar. "
        "First, think through the following retrieved verses step by step, "
        "explaining how each one contributes to your understanding of the question. "
        "Then provide a concise answer in no more than three sentences. "
        "If you do not know, simply say you don't know."
        "{context}"
    ) 
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return contextualize_q_prompt, qa_prompt

def build_lookups(vectordb: Chroma):
    """
    Returns two dicts:
      - verse_texts: verse_key -> page_content
      - verse_meta : verse_key -> metadata dict
    """
    raw = vectordb._collection.get()
    texts, metas = raw["documents"], raw["metadatas"]
    verse_texts: Dict[str, str] = {}
    verse_meta: Dict[str, dict] = {}
    for txt, md in zip(texts, metas):
        key = md.get("verse_key", f"{md['surah_id']}:{md['ayah']}")
        verse_texts[key] = txt
        verse_meta[key]   = md
    return verse_texts, verse_meta

def make_verse_lookup_tool(verse_texts: Dict[str, str]) -> Tool:
    def lookup_verse(key: str) -> str:
        k = key.strip()
        return verse_texts.get(k, f"Verse {k} not found.")
    return Tool(
        name="lookup_verse",
        func=lookup_verse,
        description=(
            "Fetch the exact English text of a verse given its key. "
            "Input: a verse key 'Surah:Ayah', e.g. '2:255'."
        )
    )

def make_metadata_tool(verse_meta: Dict[str, dict]) -> Tool:
    def metadata_query(query: str) -> str:
        """
        Expects '<field> of <Surah:Ayah>' e.g. 'revelation_place of 2:255'.
        Returns the exact metadata value.
        """
        m = re.match(r"^(\w+)\s*(?:of|for)\s*(\d+:\d+)$", query.strip(), flags=re.IGNORECASE)
        if not m:
            return ("Please use format '<field> of <Surah:Ayah>'. "
                    "Valid fields: revelation_place, revelation_order, juz_number, page_number, surah_name.")
        field, key = m.group(1), m.group(2)
        md = verse_meta.get(key)
        if not md:
            return f"No metadata found for verse {key}."
        if field not in md:
            return f"Metadata field '{field}' not available for verse {key}."
        return f"{field} of {key} is {md[field]}"
    return Tool(
        name="metadata_query",
        func=metadata_query,
        description=(
            "Use to fetch a metadata field for a given verse."
            "Input: '<field> of <Surah:Ayah>'."
        )
    )

def make_rag_tool(vectordb: Chroma) -> Tool:
    # build a history-aware retriever + QA chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    llm       = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    contextualize_q_prompt, qa_prompt = build_prompt()
    history_aware = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, qa_chain)

    def rag_query(q: str) -> str:
        resp = rag_chain.invoke({"input": q, "chat_history": []})
        answer = resp["answer"]
        context = resp["context"]
        # Display context (cited verses)
        sources = ""
        if not answer.lower().startswith("i don't know"):
            for doc in context:
                c = doc.page_content
                md = doc.metadata
                sources += f"• {md['surah_name']} ({md['surah_id']}:{md['ayah']}): {c}\n"
        return f"{answer}\n\nSources:\n{sources}"

    return Tool(
        name="rag_qa",
        func=rag_query,
        description="Ask any question about the Qur’an; returns a step-by-step reasoned answer with citations."
    )

def main():
    # Load persisted Chroma vector store
    print("🔄 Loading persisted vector store…")
    vectordb = load_vectorstore('vectordbs/quran_chroma_db')
    print("✅ Vector store loaded.")

    # Explore the vector store
    explore_vdb(vectordb)

    # Build lookups
    verse_texts, verse_meta = build_lookups(vectordb)

    # Create tools
    tools = [
        make_verse_lookup_tool(verse_texts),
        make_metadata_tool(verse_meta),
        make_rag_tool(vectordb),
    ]

    # Initialize zero-shot agent
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    prefix = """You are a Qur’an research assistant. 
    When a user refers to a verse key (e.g., 'Surah 2:255'), use the lookup_verse tool. 
    When a user requests metadata ('revelation_place of 2:255'), use the metadata_query tool. 
    Otherwise, for general questions, use the rag_qa tool, breaking the question into subquestions if needed, 
    and think through each retrieved verse step by step before answering. 
    Always use only the tools’ outputs—do not hallucinate."""
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": prefix}
    )
    
    # Interactive REPL
    print("\n🗣️  Conversational Qur’an QA with chain-of-thought (type 'exit' to quit)\n")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # Run the agent with the user input
        print("\n--- Assistant Reasoning & Answer ---")
        print(agent.run(user_input))

if __name__ == "__main__":
    main()