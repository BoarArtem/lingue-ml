from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage


from tavily import TavilyClient

import os
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

VOICING_SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
        "You are a friendly voice assistant. Your replies are read aloud by a "
        "text-to-speech engine, so they must sound like natural spoken conversation.\n"
        "\n"
        "Rules:\n"
        "- Answer the user's request directly and helpfully.\n"
        "- When you need current facts, news, prices, or anything that changes over "
        "time, use the available search tool instead of guessing.\n"
        "- When a search result is provided, base your answer ONLY on it, even if it "
        "contradicts what you previously believed. Never override fresh search data "
        "with your own prior knowledge.\n"
        "- Keep the answer SHORT: one to three sentences, ideally under 50 words. "
        "Give only the key point, not background or caveats unless asked.\n"
        "- Write plain spoken text: no markdown, no bullet points, no headings, "
        "no code blocks, no emojis, and no raw URLs.\n"
        "- Spell out numbers, symbols, and abbreviations the way a person would say "
        "them out loud.\n"
        "- If the question is unclear, ask one brief clarifying question instead of a "
        "long explanation.\n"
        "- Reply in the same language the user speaks."
        "Your answer will be used for voicing, so your response has to be clear."
     ),
    ("user", "{input}")
])

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

@tool
def search(query: str):
    """Search the web for current, factual information to answer the user's question."""
    response = tavily_client.search(
        query=query,
        include_answer=True,
    )

    # Return a clean, grounded string instead of the raw dict so a small model
    # can read it. The synthesized `answer` carries the up-to-date figure; the
    # top sources back it up.
    answer = response.get("answer") or ""
    sources = "\n".join(
        f"- {r['title']}: {r['content']}"
        for r in response.get("results", [])[:5]
    )

    return f"{answer}\n\nSources:\n{sources}".strip()


class LLMWithTools:
    def __init__(self, langchain_model_name: str = os.getenv("OLLAMA_MODEL_NAME")):
        self.tools = [search]

        self.str_output_parser = StrOutputParser()

        self.llm = ChatOllama(model=langchain_model_name)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.llm_with_history = RunnableWithMessageHistory(
            self.llm_with_tools,
            get_session_history,
        )

    def invoke(self, prompt: str, session_id: str = "user_session_1"):
        messages = VOICING_SYSTEM_PROMPT.invoke(
            {"input": prompt},
        ).to_messages()

        config = {"configurable": {"session_id": session_id}}

        ai_response = self.llm_with_tools.invoke(messages)

        for tool_call in ai_response.tool_calls:
            if tool_call["name"] == "search":
                result = search.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        final_response = self.llm_with_history.invoke(messages, config=config)

        return final_response.content

if __name__ == "__main__":
    llm = LLMWithTools(langchain_model_name="qwen2.5:7b")

    while True:
        query = input("Query: ")

        response = llm.invoke(query)

        print(response)