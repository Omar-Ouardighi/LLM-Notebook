from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agents import ResearchAgent, DataAnalystAgent
from fpl_data_tool import get_draft_players_data
from langchain_community.tools.tavily_search import TavilySearchResults
from state import AgentState


class WorkflowGraph:
    """Class to represent the workflow graph for the Fantasy Football Draft assistant."""

    def __init__(self):
        self.tavily_tool = TavilySearchResults(max_results=3)
        self.get_players_data = get_draft_players_data
        self.tools = [self.tavily_tool, self.get_players_data]
        self.graph = StateGraph(AgentState)

        # Initialize agents
        self.research_agent = ResearchAgent([self.tavily_tool])
        self.data_analyst_agent = DataAnalystAgent([self.get_players_data])
        
        # Add agent nodes to the graph
        self.graph.add_node("Researcher", self._create_node(self.research_agent))
        self.graph.add_node("DataAnalyst", self._create_node(self.data_analyst_agent))
        self.graph.add_node("call_tool", ToolNode(self.tools))
        
        # Define transitions between nodes
        self.graph.add_edge(START, "Researcher")
        self._setup_transitions()

    def _create_node(self, agent):
        return lambda state: agent.invoke(state)

    def _setup_transitions(self):
        self.graph.add_conditional_edges(
            "Researcher", self.router, {"continue": "DataAnalyst", "call_tool": "call_tool", END: END}
        )
        self.graph.add_conditional_edges(
            "DataAnalyst", self.router, {"continue": "Researcher", "call_tool": "call_tool", END: END}
        )
        self.graph.add_conditional_edges(
            "call_tool", lambda x: x["sender"], {"Researcher": "Researcher", "DataAnalyst": "DataAnalyst"}
        )

    def router(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "call_tool"
        if "FINAL ANSWER" in last_message.content:
            return END
        return "continue"

    def get_graph(self):
        return self.graph.compile()
