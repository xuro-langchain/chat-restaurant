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
from io import BytesIO
from PIL import Image
import re as _re

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


def build_multimodal_messages(system_prompt_with_context: str, history_messages: list[Any]) -> list[dict[str, Any]]:
    """
    Build a valid OpenAI chat 'messages' list preserving:
    - System message (instructions) as a true system role
    - Original conversation history as-is
    - Context (including Markdown images) as a separate user message with text+image parts

    The function extracts the <context>...</context> block from the system prompt and moves it
    into a user multimodal message. Local images are downscaled and JPEG-compressed, embedded as base64.
    """
    # 1) Split the instructions vs. context
    context_match = _re.search(r"<context>([\s\S]*?)<context/>", system_prompt_with_context)
    context_text = context_match.group(1) if context_match else ""
    system_text = _re.sub(r"<context>[\s\S]*?<context/>", "", system_prompt_with_context).strip()

    # 2) Convert LangChain messages to OpenAI message dicts, preserving order
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    converted_history: list[dict[str, Any]] = []
    for m in history_messages or []:
        role = getattr(m, "type", None)
        content = getattr(m, "content", None)
        if role in role_map and content is not None:
            converted_history.append({"role": role_map[role], "content": content})

    # 3) Build multimodal content from context_text
    # Match Markdown image links: ![alt](path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    MAX_IMAGES = 12
    TARGET_MAX_IMAGE_BYTES = 600 * 1024  # ~600KB per image
    MIN_JPEG_QUALITY = 35
    START_JPEG_QUALITY = 78

    content_parts: list[dict[str, Any]] = []
    images_added = 0
    last_idx = 0
    for match in _re.finditer(pattern, context_text):
        start, end = match.span()
        # Add text before the image (without the markdown image itself)
        pre = context_text[last_idx:start]
        if pre.strip():
            content_parts.append({"type": "text", "text": pre})

        alt_text, image_path = match.group(1), match.group(2)
        if images_added >= MAX_IMAGES:
            logger.warning("Skipping image due to MAX_IMAGES limit: %s", image_path)
            last_idx = end
            continue
        # Optionally include alt text to aid model understanding
        if alt_text.strip():
            content_parts.append({"type": "text", "text": f"Image: {alt_text}"})

        if image_path.startswith("http://") or image_path.startswith("https://"):
            content_parts.append({"type": "image_url", "image_url": {"url": image_path}})
            images_added += 1
        else:
            candidate_paths = [
                os.path.abspath(image_path),
                os.path.abspath(os.path.join(os.path.dirname(__file__), image_path)),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", image_path)),
            ]
            chosen_path: Optional[str] = None
            for p in candidate_paths:
                if os.path.exists(p):
                    chosen_path = p
                    break
            if chosen_path is not None:
                try:
                    with Image.open(chosen_path) as im:
                        if im.mode in ("RGBA", "LA"):
                            background = Image.new("RGB", im.size, (255, 255, 255))
                            background.paste(im, mask=im.split()[-1])
                            im = background
                        elif im.mode != "RGB":
                            im = im.convert("RGB")
                        max_side = 1280
                        w, h = im.size
                        scale = min(1.0, max_side / float(max(w, h)))
                        if scale < 1.0:
                            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        quality = START_JPEG_QUALITY
                        buffer = BytesIO()
                        im.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
                        while buffer.tell() > TARGET_MAX_IMAGE_BYTES and quality > MIN_JPEG_QUALITY:
                            quality = max(MIN_JPEG_QUALITY, quality - 8)
                            buffer.seek(0)
                            buffer.truncate(0)
                            im.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
                        data = buffer.getvalue()
                    mime = "image/jpeg"
                    b64 = base64.b64encode(data).decode("utf-8")
                    data_url = f"data:{mime};base64,{b64}"
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    images_added += 1
                except Exception as e:
                    logger.warning("Failed to read local image for embedding: %s (error: %s)", chosen_path, e)
            else:
                logger.warning("Local image not found. Searched candidates: %s (from markdown path: %s)", candidate_paths, image_path)
        last_idx = end

    # Trailing text after the last image
    tail = context_text[last_idx:]
    if tail.strip():
        content_parts.append({"type": "text", "text": tail})

    # 4) Assemble final OpenAI messages
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    messages.extend(converted_history)
    if content_parts:
        messages.append({"role": "user", "content": content_parts})
    return messages
