from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_txt_as_documents(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(raw_text)
    return [Document(page_content=chunk) for chunk in texts]
