import random
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
POPULATION_SIZE = 50
GENERATIONS     = 100
MUTATION_RATE   = 0.05
TOURNAMENT_SIZE = 5
DAYS_OF_WEEK    = ["Monday", "Tuesday", "Wednesday",
                   "Thursday", "Friday", "Saturday", "Sunday"]

# ── Fitness Function ───────────────────────────────────────
def calculate_fitness(chromosome: list[dict], prefs: dict) -> float:
    """Score a weekly plan — higher is better. Max possible score is 1.0."""
    if len(chromosome) != 7:
        return 0.0

    total_calories = sum(m["calories"] for m in chromosome)
    total_cost     = sum(m["cost_usd"] for m in chromosome) * prefs["servings"]
    total_protein  = sum(m["protein_g"] for m in chromosome)
    avg_calories   = total_calories / 7

    # ── Calorie score (0–0.4) ──────────────────────────────
    calorie_diff    = abs(avg_calories - prefs["calorie_target"])
    calorie_score   = max(0.0, 1.0 - calorie_diff / prefs["calorie_target"])
    calorie_score  *= 0.4

    # ── Budget score (0–0.3) ───────────────────────────────
    if total_cost <= prefs["budget_per_week"]:
        budget_score = 1.0
    else:
        over         = total_cost - prefs["budget_per_week"]
        budget_score = max(0.0, 1.0 - over / prefs["budget_per_week"])
    budget_score *= 0.3

    # ── Protein score based on health goal (0–0.2) ─────────
    goal = prefs["health_goal"]
    if goal == "gain_muscle":
        target_protein = prefs["calorie_target"] * 0.3 / 4
    elif goal == "lose_weight":
        target_protein = prefs["calorie_target"] * 0.25 / 4
    else:
        target_protein = prefs["calorie_target"] * 0.2 / 4

    protein_diff   = abs(total_protein - target_protein * 7)
    protein_score  = max(0.0, 1.0 - protein_diff / (target_protein * 7))
    protein_score *= 0.2

    # ── Variety score — penalise repeated meals (0–0.1) ────
    unique_meals   = len(set(m["id"] for m in chromosome))
    variety_score  = (unique_meals / 7) * 0.1

    total = calorie_score + budget_score + protein_score + variety_score
    return round(total, 4)


# ── Population Helpers ─────────────────────────────────────
def create_chromosome(meal_pool: list[dict]) -> list[dict]:
    """Create one random weekly plan (7 meals) from the pool."""
    return random.choices(meal_pool, k=7)


def create_population(meal_pool: list[dict]) -> list[list[dict]]:
    """Create the initial population of 50 chromosomes."""
    return [create_chromosome(meal_pool) for _ in range(POPULATION_SIZE)]


def tournament_selection(population: list, fitnesses: list[float]) -> list[dict]:
    """Pick the best chromosome from a random tournament subset."""
    indices   = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_idx  = max(indices, key=lambda i: fitnesses[i])
    return deepcopy(population[best_idx])


def crossover(parent1: list[dict], parent2: list[dict]) -> list[dict]:
    """Single-point crossover — combine two parent plans into one child."""
    point = random.randint(1, 6)
    return parent1[:point] + parent2[point:]


def mutate(chromosome: list[dict], meal_pool: list[dict]) -> list[dict]:
    """Randomly replace meals with MUTATION_RATE probability."""
    return [
        random.choice(meal_pool) if random.random() < MUTATION_RATE else meal
        for meal in chromosome
    ]


# ── Main Optimizer ─────────────────────────────────────────
def run_genetic_optimizer(meals: list[dict], prefs: dict) -> dict:
    """Run the genetic algorithm and return the best weekly plan found."""
    logger.info(f"Starting GA — population={POPULATION_SIZE}, generations={GENERATIONS}")

    # Filter meals by diet type
    meal_pool = [m for m in meals if prefs["diet_type"] in m["diet_types"]]
    if len(meal_pool) < 7:
        meal_pool = meals

    population = create_population(meal_pool)
    best_chromosome = None
    best_fitness    = -1.0

    for gen in range(GENERATIONS):
        fitnesses = [calculate_fitness(c, prefs) for c in population]

        # Track best overall
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness    = fitnesses[gen_best_idx]
            best_chromosome = deepcopy(population[gen_best_idx])

        if gen % 20 == 0:
            logger.info(f"Gen {gen:03d} — best fitness: {best_fitness:.4f}")

        # Build next generation
        next_population = [deepcopy(best_chromosome)]  # elitism — keep best
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child   = crossover(parent1, parent2)
            child   = mutate(child, meal_pool)
            next_population.append(child)

        population = next_population

    logger.info(f"GA complete — best fitness: {best_fitness:.4f}")

    # Build result
    days = [
        {"day": DAYS_OF_WEEK[i], "meal": best_chromosome[i]}
        for i in range(7)
    ]
    total_cost   = round(sum(m["cost_usd"] for m in best_chromosome) * prefs["servings"], 2)
    avg_calories = round(sum(m["calories"] for m in best_chromosome) / 7, 1)

    return {
        "days":                   days,
        "total_cost":             total_cost,
        "average_daily_calories": avg_calories,
        "fitness_score":          best_fitness,
    }