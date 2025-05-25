# Agricultural Assistant (CrewAI)

## Overview
This project is an intelligent agricultural assistant that answers farming and agriculture-related queries using both internal document search (via FAISS vector database) and web search fallback. It leverages LLMs (like Llama-3/4 via Groq or Ollama) and LangChain for retrieval-augmented generation (RAG).

## Project Structure
- `RAG.py`: Main logic for agent setup, document and web search tools, and agent execution.
- `app.py`: Streamlit web interface for user queries.
- `chunker.py`: Custom text splitter for JSON-based document chunking.
- `custom_tools.py`: Tools for embedding, vectorstore management, and web search.
- `WebSearchTool.py`: Advanced web scraping and extraction tool.
- `collection/`: FAISS vector database directory (persistent storage).
- `Datasets/`: Raw data files (CSV, etc.).
- `Preprocessed_Json_Files/`: Preprocessed JSON files ready for embedding.

## Key Files and Their Functions

### `RAG.py`
- **safe_threshold_retriever(query, threshold, k)**: Searches the FAISS vector DB for relevant documents above a similarity threshold.
- **document_search_tool**: LangChain Tool wrapping the above function for agent use.
- **analyze_and_respond_fn(context)**: Summarizes content into bullet-format Markdown using the LLM.
- **tavily_search_tool**: Web search fallback tool using TavilySearch.
- **Agent setup**: Combines tools, prompt, and memory into a ReAct agent for agricultural Q&A.

### `app.py`
- **Streamlit UI**: Presents a text area for user queries and displays answers.
- **agent_executor.invoke({"input": query})**: Sends user input to the agent and renders the response.

### `custom_tools.py`
- **EmbeddingLoader**: Loads/prepares embeddings from preprocessed JSON, manages FAISS vectorstore creation and loading.
  - `create_vectorstore()`: Embeds documents and saves/merges to FAISS DB.
  - `load_vectorstore()`: Loads an existing FAISS DB.
- **DocumentSearchTool**: Custom tool for searching the vectorstore with a query.
- **WebSearchTool**: Uses Google search and the `scrap` function to extract structured info from web pages.

### `WebSearchTool.py`
- **scrap(url)**: Asynchronously crawls a URL, extracts structured content (overview, key details, topics, tables) using an LLM extraction strategy.
- **Blog (Pydantic model)**: Defines the schema for extracted blog/article content.
- **Main block**: Example usage for crawling top Google search results.

## How to Add New Data to FAISS Vector DB
1. **Prepare Data**: Place your raw data in `Datasets/` and preprocess it into the required JSON format (see `preprocessing.ipynb`).
2. **Preprocess**: Save the preprocessed JSON in `Preprocessed_Json_Files/`.
3. **Run Merge Script**: Use the provided script (see `merge_faiss.py`) to add new vectors to the existing FAISS DB in `collection/`.

## Running the App
1. Install dependencies (see below).
2. Set your API keys as environment variables (`GROQ_API_KEY`, `TAVILY_API_KEY`).
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dependencies
- Python 3.10+
- langchain, langchain_ollama, langchain_groq, langchain_tavily
- streamlit
- faiss-cpu
- tqdm
- pydantic
- Other: see your environment or requirements.txt

## Example Usage
- Ask questions like "How to control fungus in rice crop?"
- The assistant will search internal documents first, then the web if needed, and provide a structured answer.

## License
MIT