import joblib
import logging
import numpy as np
from pathlib import Path

logger    = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).parent / "model.pkl"

_model = None

def load_model():
    """Load the model from disk — cached after first load."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            logger.warning("Model not found — returning default score")
            return None
        _model = joblib.load(MODEL_PATH)
        logger.info("ML model loaded from disk")
    return _model

def predict_quality(plan_days: list[dict], calorie_target: int = 2000,
                    budget: float = 80.0) -> float:
    """Predict quality score for a weekly plan. Returns float 0.0-1.0."""
    model = load_model()
    if model is None:
        return 0.0

    meals = [entry["meal"] for entry in plan_days]

    total_calories = sum(m["calories"]      for m in meals)
    total_cost     = sum(m["cost_usd"]      for m in meals)
    total_protein  = sum(m["protein_g"]     for m in meals)
    total_carbs    = sum(m["carbs_g"]       for m in meals)
    total_fat      = sum(m["fat_g"]         for m in meals)
    avg_prep       = sum(m["prep_time_min"] for m in meals) / 7

    calories_delta = abs((total_calories / 7) - calorie_target)
    budget_delta   = max(0, total_cost - budget)

    features = np.array([[
        calories_delta,
        budget_delta,
        total_protein,
        total_carbs,
        total_fat,
        avg_prep,
    ]])

    score = float(model.predict(features)[0])
    return round(max(0.0, min(1.0, score)), 4)