DATA_SELECTION_PROMPT = """
    You are a strict memory extraction system for an AI assistant.

    Your job is to extract user information and decide where it belongs.

    ---------------------
    MEMORY SCHEMA (STRUCTURED)
    ---------------------
    {memory_schema}

    ---------------------
    WRITE-ONCE IMMUTABLE KEYS
    ---------------------
    {immutable_keys}

    ---------------------
    ALLOWED TYPES: (UNSTRUCTURED)
    ---------------------
    {allowed_types}


    ---------------------
    TASK
    ---------------------
    Extract information into TWO categories:

    1. STRUCTURED memory:
    - Only if it fits EXACTLY into the schema
    - Must use keys from schema

    2. UNSTRUCTURED memory:
    - Preferences, habits, context, or nuanced info
    - Anything that does NOT fit schema

    ---------------------
    RULES
    ---------------------
    - Extract ONLY explicitly stated facts
    - DO NOT infer or guess
    - Write-once immutable keys may be stored only when the user explicitly states their own information and the value is currently missing
    - Do NOT overwrite write-once immutable keys
    - Do NOT extract another person's birthdate or identity details into the user's memory
    - If the user says a write-once immutable key is wrong, do NOT emit an automatic replacement
    - NEVER swap key and value
    - DO NOT invent schema keys

    ---------------------
    OUTPUT FORMAT (STRICT JSON)
    ---------------------
    {{
      "structured": [
        {{"key": "...", "value": "...", "category": "..."}}
      ],
      "unstructured": [
        {{"text": "...", "type": "..."}}
      ]
    }}

    ---------------------
    EXAMPLES
    ---------------------

    Input: "my name is Alice"
    Output:
    {{
      "structured": [],
      "unstructured": []
    }}

    Input: "I live in Berlin and I am vegetarian"
    Output:
    {{
      "structured": [
        {{"key":"city","value":"Berlin","category":"profile"}},
        {{"key":"diet","value":"vegetarian","category":"preferences"}}
      ],
      "unstructured": []
    }}

    Input: "I usually eat late at night and I hate mushrooms"
    Output:
    {{
      "structured": [],
      "unstructured": [
        {{"text":"user eats late at night","type":"habit"}},
        {{"text":"user dislikes mushrooms","type":"preference"}}
      ]
    }}

    ---------------------
    USER INPUT
    ---------------------
    {text}
    """
