"""Load html from files, clean up, split, ingest into Weaviate."""

import os
import re
import hashlib
import logging
from typing import Optional

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import os
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.embeddings import get_embeddings_model
from backend.parser import langchain_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def metadata_extractor(
    meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "")
        if description_element
        else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError(
            "Input should be either BeautifulSoup object or an HTML string"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()



def load_pdf_report(pdf_path: str, image_dir: str = "assets/images") -> list:
    docs = []
    try:
        loader = PyPDFLoader(pdf_path)
        langchain_docs = loader.load()
        from pypdf import PdfReader
        os.makedirs(image_dir, exist_ok=True)
        reader = PdfReader(pdf_path)
        page_images = {}
        # Get a safe base name for the document
        doc_base = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, page in enumerate(reader.pages):
            images = []
            if hasattr(page, "images"):
                for img_index, img in enumerate(getattr(page, "images", [])):
                    try:
                        img_data = img.data
                        img_hash = hashlib.sha1(img_data).hexdigest()[:8]
                        img_ext = img.name.split(".")[-1] if "." in img.name else "png"
                        img_filename = f"{doc_base}-p-{i+1}-i-{img_index+1}-{img_hash}.{img_ext}"
                        img_path = os.path.join(image_dir, img_filename)
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        images.append(img_path)
                    except Exception as e:
                        logger.error(f"Failed to extract image from page {i+1}: {e}")
            page_images[i] = images
        for doc in langchain_docs:
            page_num = doc.metadata.get("page", None)
            if page_num is not None:
                images = page_images.get(page_num - 1, [])
                if images:
                    for img_path in images:
                        rel_path = os.path.relpath(img_path, start=os.path.dirname(__file__))
                        doc.page_content += f"\n\n![PDF Image]({rel_path})"
            docs.append(doc)
    except Exception as e:
        logger.error(f"Error loading PDF with PyPDFLoader and extracting images: {e}")
    return docs


def ingest_docs(pdf_path: str = None):
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

        if pdf_path:
            docs_from_pdf = load_pdf_report(pdf_path)
            logger.info(f"Loaded {len(docs_from_pdf)} docs from PDF: {pdf_path}")
            docs_transformed = text_splitter.split_documents(docs_from_pdf) if docs_from_pdf else []
            if not docs_transformed:
                logger.error(f"No documents loaded from PDF: {pdf_path}. Ingestion aborted.")
                return
        else:
            docs_transformed = []

        # Ensure required metadata fields
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                doc.metadata["source"] = ""
            if "title" not in doc.metadata:
                doc.metadata["title"] = ""

        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )

        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
        logger.info(
            f"LangChain now has this many vectors: {num_vecs}",
        )


if __name__ == "__main__":
    import sys
    import glob
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if pdf_path:
        ingest_docs(pdf_path)
    else:
        pdf_files = glob.glob("assets/documents/*.pdf")
        for path in pdf_files:
            ingest_docs(path)
