import json
import random
import logging
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MEALS_PATH = BASE_DIR / "data" / "meals.json"
MODEL_PATH = Path(__file__).parent / "model.pkl"

# ── Constants ──────────────────────────────────────────────
NUM_SAMPLES    = 500
RANDOM_STATE   = 42
TEST_SIZE      = 0.2

def load_meals() -> list[dict]:
    """Load meal dataset from disk."""
    with open(MEALS_PATH, "r") as f:
        return json.load(f)

def generate_plan(meals: list[dict]) -> list[dict]:
    """Generate one random 7-day meal plan."""
    return random.choices(meals, k=7)

def compute_features(plan: list[dict], calorie_target: int = 2000,
                     budget: float = 80.0) -> dict:
    """Extract ML features from a weekly plan."""
    total_calories  = sum(m["calories"]   for m in plan)
    total_cost      = sum(m["cost_usd"]   for m in plan)
    total_protein   = sum(m["protein_g"]  for m in plan)
    total_carbs     = sum(m["carbs_g"]    for m in plan)
    total_fat       = sum(m["fat_g"]      for m in plan)
    avg_prep        = sum(m["prep_time_min"] for m in plan) / 7

    calories_delta  = abs((total_calories / 7) - calorie_target)
    budget_delta    = max(0, total_cost - budget)

    return {
        "calories_delta":  round(calories_delta, 2),
        "budget_delta":    round(budget_delta, 2),
        "protein_g":       round(total_protein, 2),
        "carbs_g":         round(total_carbs, 2),
        "fat_g":           round(total_fat, 2),
        "prep_time_min":   round(avg_prep, 2),
    }

def compute_quality_score(plan: list[dict], calorie_target: int = 2000,
                           budget: float = 80.0) -> float:
    """
    Rule-based quality score (0.0 - 1.0).
    Used as the training label — simulates human expert judgment.
    """
    total_calories = sum(m["calories"]  for m in plan)
    total_cost     = sum(m["cost_usd"]  for m in plan)
    total_protein  = sum(m["protein_g"] for m in plan)
    avg_calories   = total_calories / 7

    # Calorie score
    cal_diff    = abs(avg_calories - calorie_target)
    cal_score   = max(0.0, 1.0 - cal_diff / calorie_target) * 0.4

    # Budget score
    if total_cost <= budget:
        bud_score = 1.0 * 0.3
    else:
        bud_score = max(0.0, 1.0 - (total_cost - budget) / budget) * 0.3

    # Protein score
    target_protein = calorie_target * 0.25 / 4 * 7
    prot_diff      = abs(total_protein - target_protein)
    prot_score     = max(0.0, 1.0 - prot_diff / target_protein) * 0.2

    # Variety score
    unique        = len(set(m["id"] for m in plan))
    var_score     = (unique / 7) * 0.1

    return round(cal_score + bud_score + prot_score + var_score, 4)

def generate_dataset(meals: list[dict]) -> pd.DataFrame:
    """Generate NUM_SAMPLES training rows."""
    logger.info(f"Generating {NUM_SAMPLES} training samples...")
    records = []
    for _ in range(NUM_SAMPLES):
        calorie_target = random.randint(1500, 2800)
        budget         = random.uniform(40.0, 150.0)
        plan           = generate_plan(meals)
        features       = compute_features(plan, calorie_target, budget)
        score          = compute_quality_score(plan, calorie_target, budget)
        features["quality_score"] = score
        records.append(features)
    df = pd.DataFrame(records)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Quality score stats:\n{df['quality_score'].describe()}")
    return df

def train():
    """Train the ML model and log everything to MLflow."""
    meals = load_meals()
    df    = generate_dataset(meals)

    FEATURES = ["calories_delta", "budget_delta", "protein_g",
                "carbs_g", "fat_g", "prep_time_min"]
    TARGET   = "quality_score"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("smartmeal-quality-model")

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        mse     = mean_squared_error(y_test, y_pred)
        r2      = r2_score(y_test, y_pred)
        rmse    = np.sqrt(mse)

        # Log params and metrics
        mlflow.log_param("n_estimators",  100)
        mlflow.log_param("max_depth",     8)
        mlflow.log_param("num_samples",   NUM_SAMPLES)
        mlflow.log_param("test_size",     TEST_SIZE)
        mlflow.log_metric("mse",  round(mse,  4))
        mlflow.log_metric("rmse", round(rmse, 4))
        mlflow.log_metric("r2",   round(r2,   4))
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"MSE:  {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R2:   {r2:.4f}")

        # Save model locally
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

        # Feature importance
        importances = dict(zip(FEATURES, model.feature_importances_))
        logger.info("Feature importances:")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            logger.info(f"  {feat}: {imp:.4f}")

    logger.info("Training complete.")

if __name__ == "__main__":
    train()