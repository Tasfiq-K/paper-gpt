import yaml
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Tuple
from utils.load_config import LoadConfig

APPCFG = LoadConfig()

# Load the embedding model
embedding = HuggingFaceEmbeddings(model_name=APPCFG.embedding_model)

# load the vector database
vectordb = Chroma(
    persist_directory=APPCFG.persist_directory,
    embedding_function=embedding
)

print(f"Number of vectors in vectordb: {vectordb._collection.count()}")

# Prepare the chatbot

while True:
    question = input("\n\nEnter your question or 'q' to exit: ")
    if question.lower() == 'q':
        break

    question = "# user new question:\n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    