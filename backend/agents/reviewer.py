import json
import logging
import random

logger = logging.getLogger(__name__)

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"]

def run_reviewer(state: dict) -> dict:
    """Reviewer agent — validates planner output and builds the final weekly plan."""
    logger.info("Reviewer agent running...")

    prefs     = state["preferences"]
    meals     = state["meals"]
    raw       = state.get("planner_output", "")
    meals_map = {m["id"]: m for m in meals}

    # ── Parse planner output ───────────────────────────
    selected_ids = []
    try:
        # find the JSON array in the response
        start = raw.index("[")
        end   = raw.rindex("]") + 1
        selected_ids = json.loads(raw[start:end])
    except Exception:
        logger.warning("Could not parse planner output — falling back to random selection")

    # ── Validate each ID exists and matches diet ───────
    valid_ids = [
        id_ for id_ in selected_ids
        if id_ in meals_map
        and prefs["diet_type"] in meals_map[id_]["diet_types"]
    ]

    # ── Fill missing slots with random fallback ────────
    if len(valid_ids) < 7:
        logger.warning(f"Only {len(valid_ids)} valid meals from planner — filling gaps")
        fallback_pool = [
            m["id"] for m in meals
            if prefs["diet_type"] in m["diet_types"]
            and m["id"] not in valid_ids
        ]
        random.shuffle(fallback_pool)
        while len(valid_ids) < 7 and fallback_pool:
            valid_ids.append(fallback_pool.pop())

    selected = [meals_map[id_] for id_ in valid_ids[:7]]

    # ── Build day entries ──────────────────────────────
    days = [
        {"day": DAYS_OF_WEEK[i], "meal": selected[i]}
        for i in range(7)
    ]

    total_cost    = round(sum(m["cost_usd"] for m in selected) * prefs["servings"], 2)
    avg_calories  = round(sum(m["calories"] for m in selected) / 7, 1)

    state["final_plan"] = {
        "days":                    days,
        "total_cost":              total_cost,
        "average_daily_calories":  avg_calories,
    }

    logger.info(f"Reviewer approved plan — cost=${total_cost}, avg_cal={avg_calories}")
    return state