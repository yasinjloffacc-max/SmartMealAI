import logging
import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.workflow import build_workflow
from optimizer.genetic import run_genetic_optimizer
from ml.predict import predict_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmartMeal AI", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MEALS_PATH = Path(__file__).parent / "data" / "meals.json"
workflow   = build_workflow()

class UserPreferences(BaseModel):
    diet_type:       str
    health_goal:     str
    budget_per_week: float
    calorie_target:  int
    servings:        int

def load_meals() -> list[dict]:
    """Load the meal dataset from disk."""
    if not MEALS_PATH.exists():
        raise HTTPException(status_code=500, detail="Meal dataset not found")
    with open(MEALS_PATH, "r") as f:
        return json.load(f)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/meals")
async def get_meals():
    """Return the full meal dataset."""
    return load_meals()

@app.post("/generate-plan")
async def generate_plan(prefs: UserPreferences):
    """Generate an optimized 7-day meal plan using AI + GA + ML scoring."""
    logger.info(f"Generating plan — diet={prefs.diet_type}, goal={prefs.health_goal}")

    meals = load_meals()

    # Step 1 — AI agents
    try:
        ai_result  = workflow.invoke({"preferences": prefs.model_dump(), "meals": meals})
        ai_plan    = ai_result["final_plan"]
        logger.info("AI workflow complete")
    except Exception as e:
        logger.warning(f"AI workflow failed ({e}) — using GA only")
        ai_plan = None

    # Step 2 — Genetic algorithm
    optimized = run_genetic_optimizer(meals, prefs.model_dump())

    # Step 3 — ML quality score
    ml_score = predict_quality(
        optimized["days"],
        calorie_target=prefs.calorie_target,
        budget=prefs.budget_per_week,
    )
    logger.info(f"ML quality score: {ml_score}")

    optimized["ml_quality_score"] = ml_score
    optimized["ai_selected"]      = ai_plan is not None
    return optimized