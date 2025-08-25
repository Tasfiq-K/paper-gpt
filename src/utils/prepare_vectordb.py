from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import Type
from typing import List
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB using OpenAI embeddings.

    This class facilitates the process of loading documents, chunking them, and creating a VectorDB
    with OpenAI embeddings. It provides methods to prepare and save the VectorDB.

    Parameters:
        data_directory (str or List[str]): The directory or list of directories containing the documents.
        persist_directory (str): The directory to save the VectorDB.
        embedding_model_engine (str): The engine for OpenAI embeddings.
        chunk_size (int): The size of the chunks for document processing.
        chunk_overlap (int): The overlap between chunks.

    """

    def __init__(
            self,
            data_directory: str,
            persist_directory: str,
            embedding_model_name: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> None:
        """
        Initializes the PrepareVectorDB instance.

        params:
            data_directory (str or List[str]): The directory or list of directories containing the documents.
            persist_directory (str): The directory to save the vector DB.
            embedding_model_engine (str): The engine for OpenAI embeddings.
            chunk_size (int): The size of the chunk for document processing.
            chunk_overlap (int): The overlap between chunks.
        """

        # self.embedding_model_engine = embedding_model_engine
        self.tex_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        # self.embedding = OpenAIEmbeddings()
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)


    def __load_all_documents(self) -> List:
        """
        Load all docs from the specified directory or directories.

        Returns: 
            List: A list of loaded docs.
        """

        doc_counter = 0
        if isinstance(self.data_directory, list):
            print("Loading the uploaded documents...")
            docs = []
            doc_dirs = os.listdir(self.data_directory)
            for doc_dir in doc_dirs:
                doc_list = os.listdir(os.path.join(self.data_directory, doc_dir))
                for doc in doc_list:
                    docs.extend(PyPDFLoader(os.path.join(self.data_directory, doc_dir, doc)).load())
                    doc_counter += 1
            print(f"Number of Loadedd documents: {doc_counter}")
            print(f"Number of pages: {len(docs)}\n\n")
        else:
            print("Loading documents Manually...")
            doc_list = os.listdir(self.data_directory)
            docs = []
            for doc in doc_list:
                docs.extend(PyPDFLoader(
                    self.data_directory, 
                    doc
                ).load()
                )
                doc_counter += 1
            print(f"Number of Loadedd documents: {doc_counter}")
            print(f"Number of pages: {len(docs)}\n\n")
        
        return docs
    
    def __chunk_documents(self, docs: List) -> List:
        """
        CHunk the loaded documents using the specified text splitter.

        Params: 
            Chroma: The created VectorDB
        """
        print(f"Chunking Begins...")
        chunked_docs = self.tex_splitter.split_documents(docs)
        print(f"Number of chunks: {len(chunked_docs)}\n\n")

        return chunked_docs

    def create_vectordb(self):
        """
        Load, chunk and create a vectorDB with openAI embeddings and save it.

        returns: 
            choram vectorDB
        """

        docs = self.__load_all_documents()
        chunked_docs = self.__chunk_documents(docs)

        print(f"Preparing vectordb. Please wait...")

        vectordb = Chroma.from_documents(
            documents=chunked_docs,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
        )

        print(f"VectorDB is created and saved!")
        print(f"Number of vectors in vectorDB, {vectordb._collection.count()}\n\n")

        return vectordb   



if __name__ == "__main__":
    embedding_engine = 'BAAI/bge-m3'
    pdb = PrepareVectorDB(
        data_directory='data/docs',
        persist_directory='data/chroma_db',
        embedding_model_name=embedding_engine,
        chunk_size=1500, 
        chunk_overlap=300
    )

    pdb.create_vectordb()

    