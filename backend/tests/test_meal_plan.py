import pytest
import json
from pathlib import Path
from optimizer.genetic import calculate_fitness, create_chromosome, run_genetic_optimizer

MEALS_PATH = Path(__file__).parent.parent / "data" / "meals.json"

@pytest.fixture
def meals():
    """Load meal dataset for tests."""
    with open(MEALS_PATH) as f:
        return json.load(f)

@pytest.fixture
def sample_prefs():
    """Sample user preferences for tests."""
    return {
        "diet_type":       "omnivore",
        "health_goal":     "maintain",
        "budget_per_week": 80.0,
        "calorie_target":  2000,
        "servings":        1,
    }

def test_meals_dataset_loaded(meals):
    """Dataset must have at least 200 meals."""
    assert len(meals) >= 200

def test_meal_has_required_fields(meals):
    """Every meal must have all required fields."""
    required = {"id", "name", "calories", "protein_g", "carbs_g",
                "fat_g", "cost_usd", "prep_time_min", "diet_types", "ingredients"}
    for meal in meals:
        assert required.issubset(meal.keys()), f"Meal {meal.get('id')} missing fields"

def test_fitness_score_range(meals, sample_prefs):
    """Fitness score must be between 0 and 1."""
    chromosome = create_chromosome(meals)
    score = calculate_fitness(chromosome, sample_prefs)
    assert 0.0 <= score <= 1.0

def test_fitness_prefers_budget_compliance(meals, sample_prefs):
    """A plan within budget should score higher than one way over budget."""
    cheap_meals = [m for m in meals if m["cost_usd"] < 4.0][:7]
    expensive   = [m for m in meals if m["cost_usd"] > 10.0]
    if len(cheap_meals) < 7 or len(expensive) < 7:
        pytest.skip("Not enough meals for this test")
    expensive_7 = expensive[:7]
    cheap_score = calculate_fitness(cheap_meals, sample_prefs)
    exp_score   = calculate_fitness(expensive_7, sample_prefs)
    assert cheap_score >= exp_score

def test_genetic_optimizer_returns_7_days(meals, sample_prefs):
    """Optimizer must always return exactly 7 days."""
    result = run_genetic_optimizer(meals, sample_prefs)
    assert len(result["days"]) == 7

def test_genetic_optimizer_returns_valid_fields(meals, sample_prefs):
    """Optimizer result must contain all required fields."""
    result = run_genetic_optimizer(meals, sample_prefs)
    assert "days"                   in result
    assert "total_cost"             in result
    assert "average_daily_calories" in result
    assert "fitness_score"          in result

def test_genetic_optimizer_diet_filter(meals):
    """Optimizer must only return meals matching the requested diet type."""
    prefs = {
        "diet_type":       "vegan",
        "health_goal":     "maintain",
        "budget_per_week": 80.0,
        "calorie_target":  2000,
        "servings":        1,
    }
    result = run_genetic_optimizer(meals, prefs)
    for entry in result["days"]:
        assert "vegan" in entry["meal"]["diet_types"], \
            f"Non-vegan meal found: {entry['meal']['name']}"