from typing import Iterator
from dotenv.main import load_dotenv
from docling.document_converter import DocumentConverter

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings

import os

load_dotenv()
api_key = os.getenv("API_KEY", None)
project_id = os.getenv("PROJECT_ID", None)

embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
    EmbedParams.RETURN_OPTIONS: {
    'input_text': True
    }
}

embeddings_model = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url="https://us-south.ml.cloud.ibm.com",
    apikey=api_key,
    project_id=project_id
)

class DoclingPDFLoader(BaseLoader):

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

def pdf_processing(file_name):
    # FILE_PATH = f"/Users/chiatecksheng/Desktop/smu_workshop/docs/{file_name}"
    FILE_PATH = f"./docs/{file_name}"
    loader = DoclingPDFLoader(file_path=FILE_PATH)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    documents = []
    for chunk in chunks:
        content = chunk.page_content
        metadata = {
            "source": file_name
        }
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    return documents

def vectorstore_ingest(file_name):
    docs = pdf_processing(file_name)
    vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory="./recordb")

    return vectorstore