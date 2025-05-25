import os
import json
import asyncio
from tqdm import tqdm
from typing import Type
from googlesearch import search
from typing import Type, List
from pydantic import BaseModel, Field, ConfigDict
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from crewai.tools import BaseTool
from chunker import JSONChunkSplitter
import warnings
from WebSearchTool import scrap

warnings.filterwarnings("ignore")

class EmbeddingLoader:
    def __init__(self, file_path: str, vectorstore_path: str = "collection"):
        self.file_path = file_path
        self.vectorstore_path = vectorstore_path
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        self.splitter = JSONChunkSplitter()
        self.vectorstore = None

    def _load_json_file(self) -> List[dict]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_vectorstore(self, batch_size: int = 64):
        raw_data = self._load_json_file()
        documents = self.splitter.create_documents_from_json(raw_data)
        print(f"Total documents to embed: {len(documents)}")

        vectorstore = None

        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
            batch_docs = documents[i : i + batch_size]
            batch_vectorstore = FAISS.from_documents(
                batch_docs,
                self.embedding_model,
            )

            if vectorstore is None:
                vectorstore = batch_vectorstore
            else:
                vectorstore.merge_from(batch_vectorstore)

        self.vectorstore = vectorstore
        print("Embedding generation completed.")
        print(f"Saving vectorstore to disk at '{self.vectorstore_path}'...")
        self.vectorstore.save_local(self.vectorstore_path)

    def load_vectorstore(self):
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(
                f"Vectorstore not found at {self.vectorstore_path}, please create it first."
            )
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore


class DocumentSearchToolInput(BaseModel):
    query: str = Field(..., description="Query to search the document.")
    top_k: int = Field(
        default=5, description="Number of top similar documents to retrieve."
    )


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    model_config = ConfigDict(extra="allow")

    def __init__(self, vectorstore_path: str = "collection", top_k: int = 5):
        super().__init__()
        self.vectorstore_path = vectorstore_path
        self.embedding_loader = EmbeddingLoader(
            file_path=None, vectorstore_path=self.vectorstore_path
        )
        self.embedding_loader.embedding_model = OllamaEmbeddings(
            model="nomic-embed-text:v1.5"
        )
        self.vectorstore = self.embedding_loader.load_vectorstore()
        self.top_k = top_k

    def __repr__(self):
        return f"<DocumentSearchTool top_k={self.top_k}, vectorstore_path='{self.vectorstore_path}'>"

    def _run(self, query: str, top_k: int = 5, threshold: int = 0.4) -> str:
        print("Using the DocumentSearch Tool ...")
        try:
            results = self.vectorstore.similarity_search(query, k=self.top_k)
            if not results:
                return "No relevant documents found for the query."
            return "\n___\n".join(
                [
                    doc.page_content
                    + str(doc.metadata["Crop_info"])
                    + str(doc.metadata["geolocation"])
                    for doc in results
                ]
            )
        except Exception as e:
            print(f"[DocumentSearchTool] Error during vector search: {e}")
            return f"Error during document search: {str(e)}"


class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = "Search the Query over the Internet."

    def _run(self, query: str) -> str:
        print("Using the WebSearch Tool ...")
        try:
            urls: List[str] = [
                url
                for url in search(query, num_results=5)
                if url.startswith("https://")
            ]
            print("URLs to crawl:", urls)
        except Exception as e:
            print(f"Error during web search: {e}")
            urls = []
        return asyncio.run(scrap(urls[0]))


def test_pipeline():
    # json_path = r"Preprocessed_Json_Files\preprocessed-2024.json"

    # loader = EmbeddingLoader(file_path=json_path)
    # loader.create_vectorstore()

    # search_tool = DocumentSearchTool()
    # result = search_tool._run(
    #     query="ask about saptaparni growth and fungus attack", top_k=3
    # )
    # print("Search Results:\n", result)
    websearch = WebSearchTool()
    result = websearch._run("latest Meta LLM model")
    print("Web Search Results:\n", result)


if __name__ == "__main__":
    # test_pipeline()
    pass
