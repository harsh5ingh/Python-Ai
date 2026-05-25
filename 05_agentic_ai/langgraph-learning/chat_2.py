from dotenv import load_dotenv
import os

from typing_extensions import TypedDict
from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

class State(TypedDict):
  user_query: str
  llm_output: Optional[str]

  is_good: Optional[bool]

def chatbot(state: State):
  response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
      {"role": "user", "content": state.get("user_query")}
    ]
  )

  state["llm_output"] = response.choices[0].message.content
  return state

def evaluate_response(state: State) -> Literal["chatbot_groq", "endnode"]:
  if True:
     return "endnode"
  
  return "chatbot_groq"

def chatbot_groq(state: State):
  response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
      {"role": "user", "content": state.get("user_query")}
    ]
  )

  state["llm_output"] = response.choices[0].message.content
  return state

def endnode(state: State):
  return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chatbot_groq", chatbot_groq)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)

graph_builder.add_edge("chatbot_groq", "endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()

updated_state = graph.invoke({"user_query": "Hey, What is 2+2?"})

print("Updated State", updated_state)
