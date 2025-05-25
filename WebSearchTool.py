import os
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from googlesearch import search
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy


class Blog(BaseModel):
    title: str = Field(..., description="The title or heading of the blog or article.")
    date: Optional[str] = Field(
        None, description="The publication date of the content, if available."
    )
    detailed_description: str = Field(
        ...,
        description="A thorough summary or explanation of the content, capturing key ideas, facts, or sections.",
    )


Blog.model_rebuild()


async def scrap(url: str):
    instruction = """
Extract key information from the web content in structured markdown format. Include:
1. Overview: concise summary of main topic
2. Key details: title, publication date, author/source, key findings/statistics
3. Topics covered: main subtopics or themes
4. Structured data: relevant tables or bullet points if any

Use minimal tokens while maintaining accuracy and relevance.
Respond strictly following the Blog schema.
"""

    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            api_token=os.getenv("GROQ_API_KEY"),
            max_tokens=4000,
        ),
        schema=Blog.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=800,  # smaller chunk for better focus
        overlap_rate=0.15,  # small overlap to avoid info loss
        apply_chunking=True,
        input_format="markdown",
        extra_args={
            "temperature": 0.0,  # deterministic output
            "max_tokens": 600,  # smaller max tokens per chunk
            # "top_p": 1.0,           # you can add other params as needed
        },
    )

    crawl_config = CrawlerRunConfig(
        word_count_threshold=2000,  # smaller crawl window for efficiency
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.WRITE_ONLY,  # use caching to avoid repeat calls
    )

    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)

        if result.success:
            try:
                data = json.loads(result.extracted_content)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                print("⚠️ Could not parse extracted content as JSON.")
            llm_strategy.show_usage()
        else:
            print(f"Error crawling {url}: {result.error_message}")


if __name__ == "__main__":
    urls: List[str] = [
        url
        for url in search("latest Meta LLM model", num_results=5)
        if url.startswith("https://")
    ]
    print("URLs to crawl:", urls)

    if not urls:
        print("No URLs found. Exiting.")
        exit(0)
    asyncio.run(scrap(urls[0]))