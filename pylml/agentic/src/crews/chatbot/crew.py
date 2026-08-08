import os
from typing import Any
from crewai import Crew, LLM, Process

from ...agents import RAGAgent, OrchestratorAgent


class ChatbotCrew:
    def __init__(
        self,
        db_host: str = os.getenv("HACKUITY_API_HOST", ""),
        db_password: str = os.getenv("HACKUITY_API_TOKEN", ""),
        db_user: str = os.getenv("HACKUITY_CORP_ID", ""),
        INFERENCE_MODEL_NAME: str = os.getenv("INFERENCE_MODEL_NAME", ""),
        INFERENCE_ENDPOINT: str = os.getenv("INFERENCE_ENDPOINT", ""),
        INFERENCE_API_KEY: str = os.getenv("INFERENCE_API_KEY", ""),
        step_callback: Any = None,
        task_callback: Any = None,
        **crew_kwargs,
    ):
        self.crew_kwargs = crew_kwargs
        self.step_callback = step_callback
        self.task_callback = task_callback

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

        self.hackuity_agent = RAGAgent(
            db_host=db_host,
            db_password=db_password,
            db_user=db_user,
            llm=agent_llm,
            verbose=False,
        )

        self.orchestrator_agent = OrchestratorAgent(
            llm=chat_llm,
            verbose=False,
        )

    def kickoff(self, user_input: str):

        orchestrator_task = self.orchestrator_agent.create_chat_task(user_input)

        crew = Crew(
            agents=[self.orchestrator_agent, self.hackuity_agent],
            tasks=[orchestrator_task],
            process=Process.sequential,
            stream=True,
            task_callback=self.task_callback,
            **self.crew_kwargs,
        )

        return crew.kickoff(inputs={"user_input": user_input})
