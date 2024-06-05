import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from argparse import ArgumentParser
import torch

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_raw10k")
    parser.add_argument("--persist_dir", type=str, default="chroma_data/")
    parser.add_argument("--col_name", type=str, default="wiki10k")
    parser.add_argument("--question", type=str, default="Đền Hùng ở đâu?")
    return parser.parse_args()

def create_rag_chain(data_dir="data_raw10k", persist_dir="chroma_data/", col_name="wiki10k"):
    DATA_DIR = data_dir
    CHROMA_PATH = persist_dir
    COLLECTION_NAME = col_name

    model_name = "keepitreal/vietnamese-sbert"
    embd = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_PATH, embedding_function=embd)
    retriever = vectorstore.as_retriever(k=5)

    model_id = "vilm/vinallama-2.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, torch_dtype=torch.float16, tokenizer=tokenizer, max_new_tokens=200, device="cuda"
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template_str = "Bạn là một trợ lý với nhiệm vụ trả lời câu hỏi. Hãy sử dụng những thông tin được cung cấp để trả lời câu hỏi. Nếu bạn không biết hãy trả lời là bạn không biết. Hãy trả lời một cách ngắn gọn và xúc tích.\nThông tin: {context}\nCâu hỏi: {question}\nCâu trả lời:"

    prompt = ChatPromptTemplate.from_messages([
        ("human", prompt_template_str),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs[:1])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def create_rag_chain_raw(data_dir="data_raw10k", persist_dir="chroma_data/", col_name="wiki10k"):
    DATA_DIR = data_dir
    CHROMA_PATH = persist_dir
    COLLECTION_NAME = col_name

    model_name = "keepitreal/vietnamese-sbert"
    embd = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_PATH, embedding_function=embd)
    retriever = vectorstore.as_retriever(k=5)

    model_id = "vilm/vinallama-2.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, torch_dtype=torch.float16, tokenizer=tokenizer, max_new_tokens=200, device="cuda"
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template_str = "Bạn là một trợ lý với nhiệm vụ trả lời câu hỏi. Hãy sử dụng những thông tin được cung cấp để trả lời câu hỏi. Nếu bạn không biết hãy trả lời là bạn không biết. Hãy trả lời một cách ngắn gọn và xúc tích.\nThông tin: {context}\nCâu hỏi: {question}\nCâu trả lời:"

    prompt = ChatPromptTemplate.from_messages([
        ("human", prompt_template_str),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs[:1])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
    )
    return rag_chain

def answer(question="Đền Hùng ở đâu?", chain=None):
    answer = chain.invoke(question)
    torch.cuda.empty_cache()
    return answer, question
    
if __name__ == "__main__":
    args = get_args()
    rag_chain = create_rag_chain(args.data_dir, args.persist_dir, args.col_name)
    answer, question = answer(args.question, rag_chain)
    print(question)
    print(answer[answer.find("\nCâu trả lời:")+len("\nCâu trả lời:"):].strip())