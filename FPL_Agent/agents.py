# agents.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

class Agent:
    """Base class for an agent that can interact with tools."""
    
    def __init__(self, name: str, system_message: str, tools: list):
        self.name = name
        self.system_message = system_message
        self.tools = tools
        self.llm = ChatOpenAI(model="gpt-4o-mini")  # Language model for all agents
        self.prompt = self._build_prompt()

    def _build_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, another assistant will help where you left off."
                    " When you or another assistant finds the FINAL ANSWER, prefix it so the team knows the task is complete."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=self.system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        return prompt | self.llm.bind_tools(self.tools)

    def invoke(self, state):
        result = self.prompt.invoke(state)
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=self.name)
        return {"messages": [result], "sender": self.name}

class ResearchAgent(Agent):
    """Agent specialized in researching players using a web search tool."""
    
    def __init__(self, tools: list):
        system_message = (
            "You are a specialized research assistant focusing on Fantasy Premier League (FPL) player analysis for the upcoming week in the 2024 season."
            " Your primary task is to perform a Google search to find relevant articles on the best players to draft."
            " Prioritize information on players with the highest points potential across all positions (forward, midfielder, defender, goalkeeper)."
            " If you find articles, summarize the key stats and recommendations for players."
            " you should Return the top  players for each position (forward, midfielder, defender, goalkeeper) based on their points potential to the DataAnalyst"
            "When you receive feedback from the DataAnalyst, provide final recommendations based on the data you recieved."
        )
        super().__init__(name="Researcher", system_message=system_message, tools=tools)

class DataAnalystAgent(Agent):
    """Agent specialized in fetching and analysing data for FPL players."""
    
    def __init__(self, tools: list):
        system_message = (
            "You are a data analysis assistant for Fantasy Premier League (FPL) player performance. "
            "Use the players names provided from the researcher to fetch their stats on the 2024/2025 season."
            " Your goal is to analyze the player's points potential for the upcoming week. and challenge the researcher's findings, especially the position of the players."
            " Include their fixture details for the upcoming week and provide a concise analysis of their form and whether they are worth drafting."
            " Your recommendations must be supported by relevant stats such as goals, assists, clean sheets, injury status, and recent form." 
            " Additionally, list the players by their respective positions (e.g., forwards, midfielders, defenders, goalkeepers) and evaluate their potential impact based on both individual and team performance."

        )
        super().__init__(name="DataAnalyst", system_message=system_message, tools=tools)
