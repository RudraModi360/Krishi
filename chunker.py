from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
import json

class JSONChunkSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def create_documents_from_json(self, data: list) -> list[Document]:
        documents = []

        for entry in data:
            content = f"Query: {entry['QueryText']}\nAnswer: {entry['KccAns']}"
            metadata = entry.get("metadata", {})
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError("Use `create_documents_from_json()` instead.")
