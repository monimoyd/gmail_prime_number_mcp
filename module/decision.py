from typing import List, Optional
from modules.perception import PerceptionResult
from modules.memory import MemoryItem
from modules.model_manager import ModelManager
from dotenv import load_dotenv
from google import genai
import os
import asyncio

# Optional: import logger if available
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

model = ModelManager()


async def generate_plan(
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str] = None,
    step_num: int = 1,
    max_steps: int = 3
) -> str:
    """Generates the next step plan for the agent: either tool usage or final answer."""

    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"
    tool_context = f"\nYou have access to the following tools:\n{tool_descriptions}" if tool_descriptions else ""

    prompt = f"""
You are a reasoning-driven AI agent with access to tools and memory.
Your job is to solve the user's request step-by-step by reasoning through the problem, selecting a tool if needed, and continuing until the FINAL_ANSWER is produced. 
Once email is sent, respond with FINAL_ANSWER: DONE

Respond in **exactly one line** using one of the following formats:

- FUNCTION_CALL: tool_name|param1=value1|param2=value2
- FINAL_ANSWER: DONE

🧠 Context:
- Step: {step_num} of {max_steps}
- Memory: 
{memory_texts}
{tool_context}

🎯 Input Summary:
- User input: "{perception.user_input}"
- Intent: {perception.intent}
- Entities: {', '.join(perception.entities)}
- Tool hint: {perception.tool_hint or 'None'}

✅ Examples:
FUNCTION_CALL: find_prime_numbers|input.a=[ 3, 5, 7, 9, 11, 10, 13, 17, 21]
FUNCTION_CALL: find_largest|input.a=[3, 5, 7, 11, 13, 17]
FINAL_ANSWER: [17]

✅ Examples:
- User asks: "Send email largest prime number 17 to monimoyd@gmail.com"
  - FUNCTION_CALL: send-email|recipient_id="monimoyd@gmail.com|subject="Result of Largest Prime Number|message="Largest prime number is 17"
  - FINAL_ANSWER: Email Sent

---

📏 IMPORTANT Rules:

- 🚫 Do NOT invent tools. Use only the tools listed above. Tool description has useage pattern, only use that.
- 📄 If the question may relate to public/factual knowledge (like companies, people, places), use the `search_documents` tool to look for the answer.
- 🧮 If the question is mathematical, use the appropriate math tool.
- 🔁 Analyze that whether you have already got a good factual result from a tool, do NOT search again — summarize and respond with FINAL_ANSWER.
- ❌ NEVER repeat tool calls with the same parameters unless the result was empty. When searching rely on first reponse from tools, as that is the best response probably.
- ❌ NEVER output explanation text — only structured FUNCTION_CALL or FINAL_ANSWER.
- ✅ Use nested keys like `input.string` or `input.int_list`, and square brackets for lists.
- 💡 If no tool fits or you're unsure, end with: FINAL_ANSWER: [unknown]
- ⏳ You have 3 attempts. Final attempt must end with FINAL_ANSWER.
"""



    try:
        raw = (await model.generate_text(prompt)).strip()
        log("plan", f"LLM output: {raw}")

        for line in raw.splitlines():
            if line.strip().startswith("FUNCTION_CALL:") or line.strip().startswith("FINAL_ANSWER:"):
                return line.strip()

        return "FINAL_ANSWER: DONE"

    except Exception as e:
        log("plan", f"⚠️ Planning failed: {e}")
        return "FINAL_ANSWER: DONE"

