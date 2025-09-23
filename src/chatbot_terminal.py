import yaml
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Tuple
from utils.load_config import LoadConfig
from groq import Groq  # official Groq client

APPCFG = LoadConfig()

# Initialize Groq client with API key
client = Groq(api_key=APPCFG.groq_api_key)

# Load the embedding model
embedding = HuggingFaceEmbeddings(model_name=APPCFG.embedding_model)

# Load the vector database
vectordb = Chroma(
    persist_directory=APPCFG.persist_directory,
    embedding_function=embedding
)

print(f"Number of vectors in vectordb: {vectordb._collection.count()}")

# Chatbot loop
while True:
    question = input("\n\nEnter your question or 'q' to exit: ")
    if question.lower() == 'q':
        break

    # Retrieve documents
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs_str = "\n\n".join([str(x.page_content) for x in docs])

    # Build prompt
    prompt = f"# Retrieved content:\n{retrieved_docs_str}\n\n# User question:\n{question}"

    # Call Groq LLM
    response = client.chat.completions.create(
        model=APPCFG.llm_model,  # example model
        messages=[
            {"role": "system", "content": f"{APPCFG.llsm_system_role}"},
            {"role": "user", "content": prompt}
        ],
        temperature=APPCFG.temperature,
    )

    print("\n🤖 Response:", response.choices[0].message["content"])
