import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub


load_dotenv()


def main():
    print("Hello from vectorstore-in-memory!")

    pdf_file = "/home/ushomi/projects/python/langchains/vectorstore-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_file)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstores = FAISS.from_documents(docs, embeddings)
    vectorstores.save_local("faiss_index_react")

    new_vectorestore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstores.as_retriever(), combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke({"input": "Give me gist of ReAct in 3 sentences"})


    print(result["answer"])


if __name__ == "__main__":
    main()
