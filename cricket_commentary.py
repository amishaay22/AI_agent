from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["match_situation", "batter_on_strike", "bowler", "audience", "tone"],
    template="""
You are an expert cricket commentator. 

Write an engaging cricket commentary for {match_situation} in about 100 words.

Requirements:
- Add relevant cricketing terms naturally.
- Keep it exciting and engaging.
- Add a strong hook at the beginning.
- End with a call-to-action for the audience.
-  prompt technique :persona style:irfan pathan style
-in less than 100 words

Batter on Strike: {batter_on_strike}
Bowler: {bowler}
Tone:
{tone}

Audience:
{audience}
"""
)

# User Input
match_situation = input("Match Situation: ")
batter_on_strike = input("Batter on Strike: ")
bowler = input("Bowler : ")
tone = input("Tone: ")
audience = input("Audience: ")

# Create Prompt
final_prompt = prompt.format(
    match_situation=match_situation,
    batter_on_strike=batter_on_strike,
    bowler=bowler,
    tone=tone,
    audience=audience
)

# Generate Response
response = llm.invoke(final_prompt)

# Output
print("\nGenerated Cricket Commentary:\n")
print(response.content)


#e.g. match situation: 12 run require in 5 ball..situation,batsman onsktrike, baller