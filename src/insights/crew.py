from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool
from typing import List
from dotenv import load_dotenv
import os
from insights.tools.sql_query_tool import SQLServerQueryTool
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = LLM(
    model="openrouter/deepseek/deepseek-chat-v3-0324",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Criando a ferramenta SQL personalizada
sql_tool = SQLServerQueryTool()
file_tool = FileReadTool()

@CrewBase
class Insights():
    """Insights crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    @before_kickoff
    def before_kickoff(self, inputs):
        sql_tool._execute_query_and_save_to_csv()
    
    @agent
    def engenheiro_dados(self) -> Agent:
        return Agent(
            config=self.agents_config['engenheiro_dados'],
            verbose=True,
            llm=llm,
            tools=[file_tool],
            allow_code_execution=True
        )

    @agent
    def analista_tendencias(self) -> Agent:
        return Agent(
            config=self.agents_config['analista_tendencias'],
            verbose=True,
            llm=llm,
            tools=[file_tool],
            respect_context_window=True,
            allow_code_execution=True
        )

    @agent
    def especialista_sazonalidade(self) -> Agent:
        return Agent(
            config=self.agents_config['especialista_sazonalidade'],
            verbose=True,
            llm=llm,
            tools=[file_tool],
            respect_context_window=True,
            allow_code_execution=True
        )
        
    @agent
    def especialista_projecoes(self) -> Agent:
        return Agent(
            config=self.agents_config['especialista_projecoes'],
            verbose=True,
            llm=llm,
            tools=[file_tool],
            respect_context_window=True,
            allow_code_execution=True
        )
        
    @agent
    def analista_segmentos(self) -> Agent:
        return Agent(
            config=self.agents_config['analista_segmentos'],
            verbose=True,
            llm=llm,
            tools=[file_tool],
            respect_context_window=True,
            allow_code_execution=True
        )
        
    @agent
    def especialista_relatorios(self) -> Agent:
        return Agent(
            config=self.agents_config['especialista_relatorios'],
            verbose=True,
            llm=llm,
            tools=[file_tool],
            respect_context_window=True,
            allow_code_execution=True
        )

    @task
    def engenheiro_dados_task(self) -> Task:
        return Task(
            config=self.tasks_config['engenheiro_dados_task']
        )
    
    @task
    def analista_tendencias_task(self) -> Task:
        return Task(
            config=self.tasks_config['analista_tendencias_task'],
            context=[self.engenheiro_dados_task()]
        )
    
    @task
    def especialista_sazonalidade_task(self) -> Task:
        return Task(
            config=self.tasks_config['especialista_sazonalidade_task'],
            context=[self.engenheiro_dados_task()]
        )
    
    @task
    def especialista_projecoes_task(self) -> Task:
        return Task(
            config=self.tasks_config['especialista_projecoes_task'],
            context=[self.engenheiro_dados_task(), self.especialista_sazonalidade_task()]
        )
    
    @task
    def analista_categorias_task(self) -> Task:
        return Task(
            config=self.tasks_config['analista_categorias_task'],
            context=[self.engenheiro_dados_task()]
        )
    
    @task
    def relatorio_final_task(self) -> Task:
        return Task(
            config=self.tasks_config['relatorio_final_task'],
            context=[self.engenheiro_dados_task(), self.analista_tendencias_task(), self.especialista_sazonalidade_task(), self.especialista_projecoes_task(), self.analista_categorias_task()],
            output_file='output/relatorio_final.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Insights crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
