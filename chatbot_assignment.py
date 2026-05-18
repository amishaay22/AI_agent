
import os
import re
import random
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "Set GEMINI_API_KEY in your environment or .env file.\n"
        "Get a free key at: https://aistudio.google.com/app/apikey"
    )


_SAFE_CALC = re.compile(r"^[0-9+\-*/().%\s]+$")

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression using +, -, *, /, %, and parentheses.
    Example inputs: '347 + 892', '(12.5 * 4) / 2'
    """
    expr = expression.strip()
    if not expr or not _SAFE_CALC.fullmatch(expr):
        return "Error: Only numbers and + - * / % ( ) are allowed."
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {e}"


MOCK_USERS = {
    "42":  {"name": "Jamie Rivera",    "department": "Platform",     "role": "Senior Engineer"},
    "7":   {"name": "Sam Okonkwo",     "department": "Data Science", "role": "ML Engineer"},
    "13":  {"name": "Priya Sharma",    "department": "Product",      "role": "Product Manager"},
    "99":  {"name": "Alex Chen",       "department": "Security",     "role": "Security Analyst"},
    "101": {"name": "Fatima Al-Zahra", "department": "Design",       "role": "UX Lead"},
}

@tool
def user_lookup(user_id: str) -> str:
    """
    Look up an employee record by their numeric user ID.
    Returns name, department, and role.
    Example: user_lookup('42')
    """
    key = user_id.strip()
    row = MOCK_USERS.get(key)
    if not row:
        return f"No employee found with ID {key!r}. Valid IDs: {', '.join(MOCK_USERS)}"
    return f"ID {key}: {row['name']} — {row['role']}, {row['department']} dept."


JOKES = [
    ("Why do programmers prefer dark mode?",      "Because light attracts bugs! 🐛"),
    ("Why did the developer go broke?",            "Because they used up all their cache! 💸"),
    ("What's a computer's favourite snack?",       "Microchips! 🍟"),
    ("Why is Python so great?",                    "Because it has a lot of class! 🐍"),
    ("How do you comfort a JavaScript bug?",       "You console it! 😄"),
    ("Why did the AI break up with the database?", "Because it couldn't find a relation! 💔"),
    ("What do you call a sleeping dinosaur?",      "A dino-snore! 🦕"),
    ("Why can't a bicycle stand on its own?",      "It's two-tired! 🚲"),
]

@tool
def tell_joke(category: str = "any") -> str:
    """
    Tell a random joke. Category: 'programming', 'general', or 'any'.
    Use when the user asks for a joke or wants to laugh.
    """
    cat = category.lower()
    pool = JOKES[:6] if cat == "programming" else JOKES[6:] if cat == "general" else JOKES
    setup, punchline = random.choice(pool)
    return f"😄 {setup}\n👉 {punchline}"


TOOLS = [calculator, user_lookup, tell_joke]

SYSTEM_PROMPT = """You are a smart, friendly AI assistant with 3 tools:
- calculator  — solve maths expressions
- user_lookup — look up employees by ID (valid IDs: 7, 13, 42, 99, 101)
- tell_joke   — tell a programming or general joke

Always use the right tool. If no tool is needed, answer directly."""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else p.get("text", "") for p in content]
        return " ".join(parts).strip()
    return str(content).strip()



def run_agent(question: str, history: list):
    history.append(HumanMessage(content=question))
    result = agent.invoke({"messages": history})
    ai_msg = result["messages"][-1]
    history.append(ai_msg)
    if len(history) > 20:
        history = history[-20:]
    return extract_text(ai_msg.content), history


def main():
    print("\n" + "═"*50)
    print("  🤖  Chatbot  —  Gemini 1.5 Flash")
    print("  Tools: calculator | user_lookup | tell_joke")
    print("  Type 'quit' to exit")
    print("═"*50 + "\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! 👋")
            break

        answer, history = run_agent(user_input, history)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()