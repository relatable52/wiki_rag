import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from argparse import ArgumentParser
import torch

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw10k")
    parser.add_argument("--persist_dir", type=str, default="chroma_data/")
    parser.add_argument("--col_name", type=str, default="wiki10k")
    return parser.parse_args()

def main():
    args = get_args()
    DATA_DIR = args.data_dir
    CHROMA_PATH = args.persist_dir
    COLLECTION_NAME = args.col_name

    wiki_articles_list = os.listdir(DATA_DIR)

    model_name = "keepitreal/vietnamese-sbert"
    embd = HuggingFaceEmbeddings(model_name=model_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)

    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embd, persist_directory=CHROMA_PATH)

    print("Creating vector store...")
    loop = tqdm(wiki_articles_list)
    for i in loop:
        loop.set_description(f"Processing {i}")
        loader = TextLoader(os.path.join(DATA_DIR, i), encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
        splits = splitter.split_documents(docs)
        vectorstore.add_documents(splits)

    vectorstore.persist()

if __name__ == "__main__":
    main()