"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

import uuid
from typing import Any, Literal, Optional, Union
import re
import os
import base64
import mimetypes
import logging

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)

def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    model_kwargs = {"temperature": 0}
    if provider == "google_genai":
        model_kwargs["convert_system_message_to_human"] = True
    return init_chat_model(model, model_provider=provider, **model_kwargs)


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, or a single string.
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": str(uuid.uuid4())})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = str(uuid.uuid4())
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid", str(uuid.uuid4()))

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**item, metadata={**metadata, "uuid": item_id})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid")
                if item_id is None:
                    item_id = str(uuid.uuid4())
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list


def build_multimodal_messages(xml_context: str) -> list[dict[str, Any]]:
    """
    Build a valid OpenAI chat 'messages' list from a text context that may include Markdown images.

    Output format:
    [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "..."},
          {"type": "image_url", "image_url": {"url": "https://... or file://..."}},
          ...
        ]
      }
    ]
    """
    # Split on Markdown image links: ![alt](path)
    pattern = r'!\[[^\]]*\]\(([^)]+)\)'
    parts = re.split(pattern, xml_context)
    matches = re.findall(pattern, xml_context)

    content_parts: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if part.strip():
            content_parts.append({"type": "text", "text": part})
        if i < len(matches):
            image_path = matches[i]
            # Prefer remote URLs as-is
            if image_path.startswith("http://") or image_path.startswith("https://"):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_path},
                })
            else:
                # Embed local images as data URLs (base64) per OpenAI vision format
                candidate_paths = []
                # 1) As provided (relative to CWD)
                candidate_paths.append(os.path.abspath(image_path))
                # 2) Relative to this file's directory
                candidate_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), image_path)))
                # 3) Project root guess (two levels up from backend/)
                candidate_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", image_path)))

                chosen_path: Optional[str] = None
                for p in candidate_paths:
                    if os.path.exists(p):
                        chosen_path = p
                        break

                if chosen_path is not None:
                    try:
                        with open(chosen_path, "rb") as f:
                            data = f.read()
                        mime, _ = mimetypes.guess_type(chosen_path)
                        if not mime:
                            mime = "image/png"
                        b64 = base64.b64encode(data).decode("utf-8")
                        data_url = f"data:{mime};base64,{b64}"
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        })
                    except Exception as e:
                        logger.warning(
                            "Failed to read local image for embedding: %s (error: %s)",
                            chosen_path,
                            e,
                        )
                else:
                    logger.warning(
                        "Local image not found. Searched candidates: %s (from markdown path: %s)",
                        candidate_paths,
                        image_path,
                    )

    return [{"role": "user", "content": content_parts}]
