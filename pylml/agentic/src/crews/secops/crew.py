# crew.py
from crewai import Crew, Task, Agent
from crewai.project import CrewBase, agent, crew, task
from agents import HackuityAgent


@CrewBase
class SecOpsCrew:
    def __init__(self, api_host: str, access_token: str, corp_id: str):
        # Crew stores settings to pass downwards
        self.api_host = api_host
        self.access_token = access_token
        self.corp_id = corp_id

    @agent
    def hackuity_agent(self) -> Agent:
        return HackuityAgent(
            api_host=self.api_host,
            access_token=self.access_token,
            corp_id=self.corp_id,
            verbose=True,
        )

    @task
    def vulnerability_ingestion_task(self) -> Task:
        return Task(
            description="Collect assets and check for corresponding active findings.",
            expected_output="An inventory breakdown of assets and vulnerabilities.",
            agent=self.hackuity_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


# Execution Context
ngsra_crew = SecOpsCrew(
    api_host="https://api.hackuity.io", access_token="SECRET_TOKEN", corp_id="CORP_123"
).crew()

result = ngsra_crew.kickoff()
