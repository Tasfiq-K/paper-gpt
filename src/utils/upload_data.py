import os
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig

CONFIG = LoadConfig()

def upload_data() -> None:
    """
    Uploads data manually to the VectorDB.

    This function initializes a PrepareVectorDB instance with configuration parameters
    such as data_directory, persist_directory, embedding_model_engine, chunk_size,
    and chunk_overlap. It then checks if the VectorDB already exists in the specified
    persist_directory. If not, it calls the prepare_and_save_vectordb method to
    create and save the VectorDB. If the VectorDB already exists, a message is printed
    indicating its presence.

    Returns:
        None
    """ 

    