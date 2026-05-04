# Demo Script

This script is designed for a 3-5 minute recruiter demo in the CLI.

Start the app:

```bash
./.venv/bin/python main.py
```

## Goal

Show that the agent can:

1. save stable user memory
2. retrieve it later
3. derive useful answers from it
4. use preferences in recommendations
5. protect immutable memory
6. resolve immutable conflicts with `yes` / `no`

## Main Demo Flow

### 1. Save a birthdate

```text
You: I was born on 1995-04-12.
```

What to say:

- “The agent should store this as structured memory.”
- “Birthdate is treated as write-once immutable.”

### 2. Save food and health preferences

```text
You: I dislike pork, I prefer simple meals, and I am trying to lose weight.
```

What to say:

- “Now we have both structured and unstructured memory in play.”
- “The goal is a stable preference signal for later recommendations.”

### 3. Ask for a recommendation that should use memory

```text
You: What should I cook today?
```

What to highlight:

- the recommendation should avoid pork
- it should lean toward the weight-loss goal
- it may prefer simple meal ideas

### 4. Ask a later age question

```text
You: How old am I?
```

What to say:

- “This is the important memory test: the agent has to retrieve the stored birthdate and derive age from it.”

### 5. Trigger an immutable conflict

```text
You: Actually, my birthdate is 1996-04-12.
```

Expected behavior:

```text
AI: I currently have 1995-04-12 saved as your birthdate. Do you want me to replace it with 1996-04-12?
```

What to say:

- “The system does not silently overwrite immutable memory.”
- “It preserves the old value and asks for confirmation.”

## Branch A: Reject the change

```text
You: no
```

Expected behavior:

```text
AI: Okay, I kept your birthdate as 1995-04-12.
```

Follow-up:

```text
You: How old am I?
```

What to highlight:

- the original birthdate is still in effect

## Branch B: Accept the change

After the previous rejection, trigger the correction again:

```text
You: My birthdate is 1996-04-12.
```

Then confirm:

```text
You: yes
```

Expected behavior:

```text
AI: Got it - I updated your birthdate to 1996-04-12.
```

Follow-up:

```text
You: How old am I?
```

What to highlight:

- the confirmed value is now applied
- the confirmation resolver is explicit and safe

## Optional Talking Points

- “Structured memory writes produce decision results like `stored`, `no_change`, `ignored`, and `needs_confirmation`.”
- “Structured memory decisions are logged locally to JSONL for debugging.”
- “Planner and ranker outputs are normalized so malformed LLM output cannot dump all memory into context.”
- “The critical memory and retrieval paths are covered by offline tests.”
