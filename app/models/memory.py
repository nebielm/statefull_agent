ALLOWED_KEYS = {
    "weight",
    "target_weight",
    "location",
    "goal",
    "balance",
    "calories",
    "height",
    "age",
}

IMMUTABLE_KEYS = ["name", "birthdate", "country of origin", "skin type"]

MEMORY_SCHEMA = {
    "profile": ["city", "job", "education"],
    "preferences": ["favorite_food", "hobbies", "diet"],
    "health": ["allergies"],
    "household": ["household_size"],
    "kitchen": ["appliances"],
    "dynamic": ["current_goal", "mood", "weight"],
}

ALLOWED_TYPES = ["habit", "preference", "diet", "dislike", "behavior", "context"]
