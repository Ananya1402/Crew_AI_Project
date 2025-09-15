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

class Content(BaseModel):
    content_type: str = Field(..., description="Type of content to be created, e.g., blog post, social media post, video script")
    topic: str = Field(..., description="The main topic or theme of the content")
    target_audience: str = Field(..., description="The intended audience for the content")
    tags: list[str] = Field(..., description="Relevant tags or keywords associated with the content")
    content: str = Field(..., description="The content itself")

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
    
    @task
    def market_research(self)-> Task:
        return Task(
            config=self.tasks_config['market_research'],
            agent = self.head_of_marketing()
        )
    
    @task
    def prepare_marketing_strategy(self)-> Task:
        return Task(
            config=self.tasks_config['prepare_marketing_strategy'],
            agent = self.head_of_marketing()
        )
    
    @task
    def create_content_calendar(self) -> Task:
        return Task(
            config=self.tasks_config['create_content_calendar'],
            agent=self.content_creator_social_media()
        )

    @task
    def prepare_post_drafts(self) -> Task:
        return Task(
            config=self.tasks_config['prepare_post_drafts'],
            agent=self.content_creator_social_media(),
            output_json=Content
        )

    @task
    def prepare_scripts_for_reels(self) -> Task:
        return Task(
            config=self.tasks_config['prepare_scripts_for_reels'],
            agent=self.content_creator_social_media(),
            output_json=Content
        )

    @task
    def content_research_for_blogs(self) -> Task:
        return Task(
            config=self.tasks_config['content_research_for_blogs'],
            agent=self.content_writer_blogs()
        )

    @task
    def draft_blogs(self) -> Task:
        return Task(
            config=self.tasks_config['draft_blogs'],
            agent=self.content_writer_blogs(),
            output_json=Content
        )

    @task
    def seo_optimization(self) -> Task:
        return Task(
            config=self.tasks_config['seo_optimization'],
            agent=self.seo_specialist(),
            output_json=Content
        )
    
    @crew
    def marketing_crew(self) -> Crew:
        return Crew(
            agents = self.agents,  #although self.agents is not defined, the decorator @agent takes care of it
            tasks = self.tasks,   #although self.tasks is not defined, the decorator @task takes care of it
            process = Process.sequential,  #Tasks will be executed in the order they are defined
            verbose = True,
            planning = True,  #Enables task planning and delegation among agents
            planning_llm=llm,
            max_rpm=3
        )
    

if __name__ == "__main__":
    from datetime import datetime

    inputs = {
        "product_name": "AI Powered Excel Automation Tool",
        "target_audience": "Small and Medium Enterprises (SMEs)",
        "product_description": "A tool that automates repetitive tasks in Excel using AI, saving time and reducing errors.",
        "budget": "Rs. 50,000",
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
    crew = MarketingCrew()
    crew.marketing_crew().kickoff(inputs=inputs)
    print("Marketing crew has been successfully created and run.")