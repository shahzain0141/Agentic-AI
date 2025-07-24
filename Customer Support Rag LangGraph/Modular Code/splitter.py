from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    return [Document(page_content=chunk) for chunk in texts]
