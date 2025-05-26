from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

# load the environment variables
load_dotenv()

# class to create a vector embeddings chroma database
class CreateDatabase:

    # constructor method
    def __init__(self):
        self.CHROMA_PATH = "chroma"
        self.DATA_PATH = "data"

    # method to load the documents from the data sources
    def load_documents(self) -> list[Document]:
        loader = DirectoryLoader(self.DATA_PATH, glob="*.md")
        documents = loader.load()
        return documents


    # method to split the documents into chunks
    def text_split(self, documents: list[Document]):
        # defining the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )

        # splitting the documents into chunks
        chunks = text_splitter.split_documents(documents=documents)

        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

        return chunks


    # method to save the chunks to Chroma Data Base
    def save_to_chroma(self, chunks):
        # clear out the database if it already exists
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)

        db = Chroma.from_documents(
            chunks,
            embedding=OllamaEmbeddings(model="mxbai-embed-large"),
            persist_directory=self.CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} to {self.CHROMA_PATH} successfully")

    # method to generate the vector store database
    def generate_data_store(self):
        documents = self.load_documents()
        chunks = self.text_split(documents)
        self.save_to_chroma(chunks)

# main stats here
if __name__ == "__main__":
    obj = CreateDatabase()
    obj.generate_data_store()