import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from colorama import Fore

import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def get_weather(city: str):
    
    if not city:
       return "City name is missing"
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"

    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong"

available_tools = {
   "get_weather": get_weather
}

SYSTEM_PROMPT = """
   You're an expert AI Assistant in resolving user queries using chain of thoughts.
   You work on START, PLAN and OUTPUT steps.
   You need to first PLAN what needs to be done. The PLAN can be multiple steps.
   Once you think enough PLAN has been done, finally you can give an OUTPUT.
   You can also call a tool if required from the list of available tools.
   for every tool vall wait for the observe step which is the output from the called tool.


   Rules:
   - Strictly follow the given JSON output format
   - Only run one step at a time.
   - The sequence of steps is START ( where user gives an input), PLAN (which is going to the displayed to the user).
   - User is on Windows.

   Output JSON format:
   { "step": "START" | "PLAN" | "OUTPUT" | "TOOL" | "OBSERVE", "content": "string", "tool": "string", "input": "string", "output": "string" }

   Available Tools:
   - get_weather(city: str): Takes city name as an input string and returns the weather info about the city.

   Example 1:
   START: Hey, can you solve 2 + 3 * 5 / 10
   PLAN: {"step": "PLAN": "content": "Seems like user is interested in math problem" }
   PLAN: {"step": "PLAN": "content": "looking at the problem, we should solve this using BODMAS method"}
   PLAN: {"step": "PLAN": "content": "Yes, The BODMAS is correct thing to be done here"}
   PLAN: {"step": "PLAN": "content": "first we must multiply 3 * 5 which is 15"}
   PLAN: {"step": "PLAN": "content": "Now the equation is 2 + 15 / 10 "}
   PLAN: {"step": "PLAN": "content": "We must perform divide that is 15 / 10 = 1.5"}
   PLAN: {"step": "PLAN": "content": "Now the new equation is 2 + 1.5"}
   PLAN: {"step": "PLAN": "content": "Now finally lets perform the add 3.5"}
   PLAN: {"step": "PLAN": "content": "Great, we have solved and finally left with 3.5 as ans"}
   PLAN: {"step": "PLAN": "content": "3.5"}

   Example 2:
   START: What is the weather of Delhi?
   PLAN: {"step": "PLAN": "content": "Seems like user is interested in getting weather of Delhi in India" }
   PLAN: {"step": "PLAN": "content": "Lets see if we have any available tool from the list of available tools"}
   PLAN: {"step": "PLAN": "content": "Great, we have get_weather tool available for this query"}
   PLAN: {"step": "PLAN": "content": "I need to call get_weather tool for delhi as input for city"}
   PLAN: {"step": "TOOL": "tool": "get_weather", "input": "Delhi"}
   PLAN: {"step": "OBSERVE": "tool": "get_weather", "output": "The temperature of Delhi is cloudy with 20 C"}
   PLAN: {"step": "PLAN": "content": "Great, I got the weather info about Delhi"}
   PLAN: {"step": "PLAN": "content": "The current weather in Delhi is 20 C with some cloudy sky"}

"""

print("\n")

message_history = [
  {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
  user_query = input("👉🏻 Enter your question:   ")
  message_history.append({"role": "user", "content": user_query})

  while True:
    response = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      response_format={"type": "json_object"},
      messages= message_history
    )

    raw_result = (response.choices[0].message.content)
    message_history.append({"role": "assistant", "content": raw_result })

    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START":
      print("🚀 Starting LLM Loop:  ", parsed_result.get("content"))
      continue

    if parsed_result.get("step") == "TOOL":
      tool_to_call = parsed_result.get("tool")
      tool_input = parsed_result.get("input")
      print(f"⚒️: {tool_to_call} ({tool_input})") 

      tool_response = available_tools[tool_to_call](tool_input)
      print(f"⚒️: {tool_to_call} ({tool_input})  = {tool_response}")
      message_history.append({
      "role": "assistant",
      "content": json.dumps({
          "step": "OBSERVE",
          "tool": tool_to_call,
          "input": tool_input,
          "output": tool_response
      })
  })
      continue



    if parsed_result.get("step") == "PLAN":
      print("🧠 Thinking...  ", parsed_result.get("content"))

    if parsed_result.get("step") == "OUTPUT":
      print(Fore.GREEN + "🤖 Output:  ", parsed_result.get("content"))
      break

    print("\n")