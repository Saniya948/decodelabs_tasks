# Project 1: Rule-Based AI Chatbot

**Author:** Saniya Inam
**Internship:** DecodeLabs — AI Track (Batch 2026)
**Track:** Artificial Intelligence Engineer, Industrial Training Kit

## Overview

This project is a simple **rule-based chatbot** built in Python. It responds to a fixed set of predefined user inputs using explicit control flow and decision-making logic — no machine learning or "deep learning" involved. The goal is to demonstrate the foundational logic (control flow, sanitization, lookup, fallback handling) that underlies more advanced generative AI systems.

## How to Run

1. Make sure Python 3 is installed (`python --version` to check).
2. Open a terminal in the project folder.
3. Run:
   ```
   python rule_based_chatbot.py
   ```
4. Chat with RuleBot! Try inputs like `hello`, `joke`, `help`, `who are you`, or `bye` to exit.

## Design Choices

**Dictionary lookup instead of if-elif chains**
Rather than a long `if / elif / elif / ...` ladder (which has O(n) lookup time and becomes harder to maintain as more rules are added), this chatbot stores all intents and responses in a single dictionary. Python dictionaries use hashing internally, giving near-constant time O(1) lookups regardless of how many rules exist. This is both faster and easier to extend — adding a new response is just one new key-value pair.

**Input sanitization**
User input is lowercased and stripped of leading/trailing whitespace before lookup. This ensures `"Hello"`, `"HELLO"`, and `" hello "` are all treated identically, rather than requiring separate rules for each variation.

**Fallback handling**
Instead of a single static "I don't understand" message, unmatched input triggers a randomly selected clarifying follow-up question. This nudges the user toward a recognized command (e.g. suggesting `help`, `joke`, or `bye`) rather than dead-ending the conversation.

**Continuous loop with clean exit**
The chatbot runs inside a `while True` loop, acting as its "heartbeat" — it stays active indefinitely until the user issues an exit command (`bye`, `exit`, or `quit`), at which point the loop breaks cleanly.

## Project Structure

```
rule_based_chatbot.py   # main chatbot script
README.md                # this file
```

## Key Concepts Demonstrated

- Control flow and decision-making logic
- Data structures (dictionaries) for efficient exact-match lookup
- Input normalization / sanitization
- Fallback and error handling
- Continuous program loops with defined exit conditions

## Possible Future Extensions

- Expand vocabulary further with more intents
- Add basic keyword/substring matching instead of exact-match only
- Layer a hybrid architecture on top: if no rule matches, pass the query to a generative model (LLM) instead of a static fallback — combining rule-based reliability with generative flexibility
