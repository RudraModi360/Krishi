import os
import warnings
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = "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"

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=1000,
)

# llm = OllamaLLM(model="llama3", temperature=0.0, streaming=True)

embeddings_obj = OllamaEmbeddings(model="nomic-embed-text:v1.5")
db = FAISS.load_local(
    "collection", embeddings_obj, allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

def safe_threshold_retriever(query: str, threshold: float = 0.4, k: int = 2) -> str:
    try:
        results = db.similarity_search_with_score(query, k=k)
        filtered = [doc for doc, score in results if score >= threshold]
        if not filtered:
            return "No relevant documents found in internal collection."
        return "\n".join(doc.page_content for doc in filtered)
    except Exception as e:
        return f"Document search failed: {str(e)}"

document_search_tool = Tool(
    name="DocumentSearch",
    func=lambda q: safe_threshold_retriever(q),
    description="Use this to search internal documents first.",
)

def analyze_and_respond_fn(context: str) -> str:
    prompt = f"Analyse this:\n{context}\n\nGive the visually appealing key points in bullet format with headings."
    return llm.invoke(prompt)

analyze_and_respond_tool = Tool(
    name="AnalyzeAndRespond",
    func=analyze_and_respond_fn,
    description="Summarizes content into bullet-format Markdown. Use only for retrieved content.",
)

def safe_tavily_search(query: str) -> str:
    try:
        result = TavilySearch(name="tavilySearch", max_results=5).run(query)
        if not result or "error" in str(result).lower():
            return "Tavily search failed or returned no result."
        return result
    except Exception as e:
        return f"Tavily search error: {str(e)}"

tavily_search_tool = Tool(
    name="tavilySearch",
    func=safe_tavily_search,
    description="Web search fallback. Use only if DocumentSearch has no results.",
)

tools = [
    document_search_tool,
    tavily_search_tool,
]

template = """
You are an intelligent agricultural assistant.

🎯 Goal:
Help users by searching internal farm documents first, and use the web as a backup if needed. Always summarize clearly.

🛠️ Tools:
{tools}

🔍 Tool Usage Guide:
- Try DocumentSearch first.
- If it fails (no useful results) or any other error, use tavilySearch.
- Only use each tool once. Avoid repeating failed tools.
- If nothing is found, clearly say so.

🧠 Thought Process Format (ReAct):
---
Question: {input}
Thought: Decide {tools} will help you?
Action: {tool_names}
Action Input: Add proper Query for this
Observation: result from tool

If Once the DocumentSearch Tools Fails Directly Shifts to other tool
(Repeat it 2-3 times)

Thought: I now have enough information to answer
Final Answer:
Not Fix for all types of Prompt but Use Commonly on Detailed Agricultural tasks ,

- **Overview**: ...
- **Control Methods**: ...
- **Best Practices**: ...
- **TNAU / ICAR Recommendations**: ...

else give Bullet wise points in final answer.

- Point1 
- Point2
---

⚠️ If no relevant info is found, clearly say: "No sufficient information found from documents or web."

Begin!
Question: {input}
Thought:{agent_scratchpad}
"""
prompt_react = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_react)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=False,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=60,
    early_stopping_method="force",
)

warnings.filterwarnings("ignore")
