import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from search import load_config, get_embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    cfg = load_config()
    embeddings = get_embeddings(cfg)

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=cfg.collection_name,
        connection=cfg.conn_str,
        pre_delete_collection=True,
        use_jsonb=True,
    )

    print("Ingestão concluída!")
    print(f"   - PDF: {PDF_PATH}")
    print(f"   - Páginas: {len(pages)}")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Coleção: {cfg.collection_name}")

if __name__ == "__main__":
    ingest_pdf()