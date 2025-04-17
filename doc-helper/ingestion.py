import os

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv(override=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def injest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(raw_documents)

    print(f"Created {len(docs)} chunks.")

    for doc in docs:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "http:/")
        doc.metadata.update({"source": new_url})

    vector_store = PineconeVectorStore(embedding=embeddings, index_name="langchain-doc-index")

    for i in range(0, len(docs), 100):
        print(f"Loading docs {i}:{i+100}")
        vector_store.add_documents(docs[i:i+100])

    print("Done loading ...")


def injest_docs_v2():
    from langchain_community.document_loaders import FireCrawlLoader

    doc_base_urls = [
        "https://python.langchain.com/docs/how_to/tool_calling/",
        "https://python.langchain.com/docs/how_to/structured_output/",
        "https://python.langchain.com/docs/how_to/chat_model_caching/",
    ]

    for url in doc_base_urls:
        print(f"FireCrawling: {url}")
        loader = FireCrawlLoader(
            api_key=os.environ["FIRECRAWL_API_KEY"],
            url=url, 
            mode="scrape",
        )

        docs = loader.load()

        print(f"Adding {len(docs)} to Pinecone.")
        PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name="firecrawl-cat-index")
        print(f"Loading of {url} done.")


def main():
    injest_docs_v2()

if __name__ == "__main__":
    main()