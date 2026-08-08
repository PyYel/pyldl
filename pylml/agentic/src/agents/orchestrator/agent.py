from typing import Any, ClassVar
from crewai import Agent, Task


class OrchestratorAgent(Agent):
    DEFAULT_ROLE: ClassVar[str] = "Assistant Chatbot"
    DEFAULT_GOAL: ClassVar[str] = (
        "Interact conversationally with the user. Answer general questions directly. "
    )
    DEFAULT_BACKSTORY: ClassVar[str] = (
        "You coordinate specialized agents and present synthesized responses to the user."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            role=self.DEFAULT_ROLE,
            goal=self.DEFAULT_GOAL,
            backstory=self.DEFAULT_BACKSTORY,
            allow_delegation=True,
            **kwargs,
        )

    def create_chat_task(self, user_input: str) -> Task:
        return Task(
            description=(
                f"Respond to the user's input: '{user_input}'\n\n"
                "Instructions:\n"
                "1. If this is casual conversation or a general question, answer directly.\n"
                "2. If the query requires specific data. "
                "delegate a task to your coworkers.\n"
            ),
            expected_output="A helpful, conversational response addressing the user's query.",
            agent=self,
        )
