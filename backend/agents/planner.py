import random
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a meal planning expert. Given the user's preferences below,
select 7 meals from the provided meal list — one per day — that best match their needs.

User preferences:
- Diet type: {diet_type}
- Health goal: {health_goal}
- Weekly budget: ${budget}
- Daily calorie target: {calories} kcal
- Servings: {servings}

Available meals (JSON list):
{meals}

Return ONLY a valid JSON array of exactly 7 meal IDs like this:
["001", "002", "003", "004", "005", "006", "007"]

Rules:
- Only pick meals that match the diet type
- Try to stay within the weekly budget
- Vary the meals — do not repeat the same meal twice
- Prefer meals whose calories are close to the daily target
Return only the JSON array, no explanation."""

def run_planner(state: dict) -> dict:
    """Planner agent — selects 7 meal IDs from the dataset using Groq."""
    logger.info("Planner agent running...")

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=256,
    api_key=os.getenv("GROQ_API_KEY"),
    )

    prefs = state["preferences"]
    meals = state["meals"]

    filtered = [
    m for m in meals
    if prefs["diet_type"] in m["diet_types"]
]
    random.shuffle(filtered)
    meal_summary = [
    {
        "id": m["id"],
        "name": m["name"],
        "calories": m["calories"],
        "cost_usd": m["cost_usd"],
        "diet_types": m["diet_types"],
        "protein_g": m["protein_g"],
        "prep_time_min": m["prep_time_min"],
    }
    for m in filtered[:30]
    ]

    prompt = PLANNER_PROMPT.format(
        diet_type=prefs["diet_type"],
        health_goal=prefs["health_goal"],
        budget=prefs["budget_per_week"],
        calories=prefs["calorie_target"],
        servings=prefs["servings"],
        meals=meal_summary,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    logger.info(f"Planner raw response: {raw}")

    state["planner_output"] = raw
    return state