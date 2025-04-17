import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


def main():
    print("Hello from intro-to-vector-db!")

    loader = TextLoader("mediumblog1.txt")
    document = loader.load()
    print("Loaded text file")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print("Split text into chunks")
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("Done")

if __name__ == "__main__":
    main()
