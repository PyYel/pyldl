from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import os
import sys
from dotenv import load_dotenv

MODULE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MODULE_DIR_PATH)

load_dotenv(
    os.path.join(MODULE_DIR_PATH, ".env"),
    override=False,
)

from .agent import RAGAgent

INFERENCE_MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "")
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")


agent_llm = LLM(
    model=INFERENCE_MODEL_NAME,
    base_url=INFERENCE_ENDPOINT,
    api_key=INFERENCE_API_KEY,
    temperature=0.7,
    provider="openai",
    max_tokens=2048,
    stream=True,
)

chat_llm = LLM(
    model=INFERENCE_MODEL_NAME,
    base_url=INFERENCE_ENDPOINT,
    api_key=INFERENCE_API_KEY,
    temperature=0.7,
    provider="openai",
    max_tokens=2048,
    stream=True,
)

agent = RAGAgent(
    db_host="",
    db_password="",
    db_user="",
    # mcp_urls=[mcp_url],
    llm=agent_llm,
    verbose=False,
    max_iter=5,
    max_execution_time=120,
)

chat_task = Task(
    description=(
        "Listen to the user's latest query: '{user_input}'. "
        "Use tools if required to complete the query, then provide a helpful response."
        "Make queries using ID, but answer the user using the object names if available."
    ),
    expected_output=(
        "A conversational and accurate answer to the user query. Reduce the volume of data and synthetize results. "
        "If response is too long (more than 20 or so items), make a summary and offer to fetch details using subqueries."
        "Always answer with the count of items found, processed, and returned."
    ),
    agent=agent,
)

chat_crew = Crew(
    agents=[agent],
    tasks=[chat_task],
    process=Process.sequential,
    chat_llm=chat_llm,
    stream=True,
)


def start_chatbot():
    print("=== Interactive CrewAI Chatbot Started ===")
    print("Type 'exit' or 'quit' to terminate the session.\n")

    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Ending session. Goodbye!")
                break

            if not user_input:
                continue

            print("\nAssistant: ", end="", flush=True)

            # 3. Iterate over the stream output chunks in real-time
            streaming_output = chat_crew.kickoff(inputs={"user_input": user_input})

            for chunk in streaming_output:
                if chunk.content:
                    print(chunk.content, end="", flush=True)

            print("\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break


if __name__ == "__main__":
    start_chatbot()
