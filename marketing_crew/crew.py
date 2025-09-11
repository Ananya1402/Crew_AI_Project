from crewai import Agent, Process, Crew, Task, LLM
from crewai.project import agent, task, crew, CrewBase
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool, FileReadTool, FileWriterTool

from pydantic import BaseModel, Field
from dotenv import load_dotenv

_ = load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7
)

@CrewBase
class MarketingCrew():
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def head_of_marketing(self) -> Agent:
        return Agent(
            config=self.agents_config['head_of_marketing'],
            tools=[SerperDevTool(), ScrapeWebsiteTool(), DirectoryReadTool('resources/drafts'), FileReadTool(), FileWriterTool()],
            reasoning=True,   #ReAct style reasoning
            inject_date=True,  #Injects current date into the prompt
            llm=llm,
            allow_delegation=True,  #this means agent can assign tasks to other agents
            max_rpm=3  #requests per minute
        )
    
    @agent
    def content_creator_social_media(self) -> Agent:
        return Agent(
            config=self.agents_config['content_creator_social_media'],
            tools=[SerperDevTool(), ScrapeWebsiteTool(), DirectoryReadTool('resources/drafts'), FileReadTool(), FileWriterTool()],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=30,
            max_rpm=3
        )
    
    @agent
    def content_writer_blogs(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer_blogs'],
            tools=[SerperDevTool(), ScrapeWebsiteTool(), DirectoryReadTool('resources/drafts/blogs'), FileReadTool(), FileWriterTool()],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=5,
            max_rpm=3
        )

    @agent
    def seo_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_specialist'],
            tools=[SerperDevTool(), ScrapeWebsiteTool(), DirectoryReadTool('resources/drafts'), FileReadTool(), FileWriterTool()],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=3,
            max_rpm=3
        )
        
