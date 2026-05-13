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
    temperature=1.0
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["customer_name", "content", "tone", "audience"],
    template="""
You are an expert social media content writer.

Write an engaging LinkedIn post for {customer_name} in about 100 words.

Requirements:
- Add relevant emojis naturally.
- Keep it professional and engaging.
- Add a strong hook at the beginning.
- End with a call-to-action.

Content:
{content}

Tone:
{tone}

Audience:
{audience}
"""
)

# User Input
customer_name = input("Customer Name: ")
content = input("Post Content: ")
tone = input("Tone: ")
audience = input("Audience: ")

# Create Prompt
final_prompt = prompt.format(
    customer_name=customer_name,
    content=content,
    tone=tone,
    audience=audience
)

# Generate Response
response = llm.invoke(final_prompt)

# Output
print("\nGenerated LinkedIn Post:\n")
print(response.content)