import os

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv(override=True)


def run_llm(query: str, chat_history: list[tuple]) -> dict:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Using index: {os.environ["INDEX_NAME"]}")

    doc_search = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    chat = ChatOpenAI(model="gpt-4.1-nano", verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm=chat, retriever=doc_search.as_retriever(), prompt=rephrase_prompt)

    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)

    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    return {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }


if __name__ == "__main__":
    res = run_llm("What is a LangChain Chain?")
    
    print(res["result"])
