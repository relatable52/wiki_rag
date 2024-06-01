import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATA_DIR = "data_raw10k"
CHROMA_PATH = "chroma_data/"

model_name = "keepitreal/vietnamese-sbert"
embd = HuggingFaceEmbeddings(model_name=model_name)

vectorstore = Chroma(collection_name="wiki10k", persist_directory=CHROMA_PATH, embedding_function=embd)
retriever = vectorstore.as_retriever()

model_id = "vilm/vinallama-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt_template_str = """Bạn là một trợ lý với nhiệm vụ trả lời câu hỏi. Hãy sử dụng những thông tin được cung cấp để trả lời câu hỏi. Nếu bạn không biết hãy trả lời là bạn không biết. Hãy trả lời một cách ngắn gọn và xúc tích.
Thông tin: {context}
Câu hỏi: {question}
Câu trả lời:"""

prompt = ChatPromptTemplate.from_template(prompt_template_str)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs[:1])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
