from dotenv import load_dotenv

from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()

llm = init_chat_model(
   model="llama-3.1-8b-instant",
   model_provider="groq"
)

class State(TypedDict):
  messages: Annotated[list, add_messages]

def chatbot(state: State):
  response = llm.invoke(state.get("messages"))
  return {
        "messages": [response]
    }

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def compile_graph_with_checkpointe(checkpointer):
  return graph_builder.compile(checkpointer=checkpointer)


DB_URI = "mongodb://admin:admin@localhost:27017/lg?authSource=admin"

with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:

  graph_With_checkpointer = compile_graph_with_checkpointe(checkpointer=checkpointer)

  config = {
          "configurable": {
              "thread_id": "harsh" # user_id
          }
  }

  for chunk in graph_With_checkpointer.stream(
  {"messages": ["Hola, Myself Harsh & I'm learning Langgraph in AI"]},
  config=config,
  stream_mode="values"
  ):
    chunk["messages"][-1].pretty_print()

# (START) -> chatbot -> samplenode -> (END)

# state = { messages: ["Hey there!"]}
# node runs: chatbot(state: ["Hey There"]) -> ["Hi, This is a message from Chatbot"]
#state = { "messages": ["Hey there", "Hi, This is a message from chatbot Node"]}

# in Checkpointer (harsh) = Hey, My name is Harsh Singh