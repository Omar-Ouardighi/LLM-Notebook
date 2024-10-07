import streamlit as st
from graph import WorkflowGraph
from langchain_core.messages import HumanMessage


logo_path = "logo.png"
col1, col2 = st.columns([1, 8])  


with col1:
    st.image(logo_path, width=80)  

with col2:
    st.title("FPL Agent")


def display_agent_messages(agent_name, messages):
    """Helper function to display agent messages with markdown."""
    if messages:
        last_message = messages[-1].get("content", "No content available.")
        with st.expander(f"{agent_name}'s Response", expanded=True):
            st.markdown(f"**{agent_name}:** \n{last_message}")


workflow_graph = WorkflowGraph()
graph = workflow_graph.get_graph()

input_message = st.text_input("Enter your request:", "draft the best players for the upcoming week in the 2024 season, week 7")

if st.button("Submit"):
    if not input_message:
        st.warning("Please enter your request.")
    else:
        with st.spinner("Agents are working ... "):
            
            events = graph.invoke({"messages": [HumanMessage(content=input_message)]}, {"recursion_limit": 50})
            st.markdown(events['messages'][-1].content)

