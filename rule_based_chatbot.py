"""
Project 1: Rule-Based AI Chatbot 
DecodeLabs - AI Internship 2026
Author: Saniya Inam

Goal: A simple rule-based chatbot that responds to predefined user
inputs using dictionary-based (O(1)) lookup instead of a long if-elif
ladder (O(n)).

"""


import random

# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
# Dictionary mapping cleaned user intents -> bot responses.
# Easy to expand: just add more key-value pairs here.
responses = {
    "hello": "Hi there! How can I help you today? 😊",
    "hi": "Hello! What can I do for you? 👋",
    "hey": "Hey! Good to see you. 😄",
    "how are you": "I'm just a program, but I'm running smoothly! How about you? 🤖",
    "what is your name": "I'm RuleBot, your friendly rule-based assistant. 🤖",
    "who are you": "I'm RuleBot — built to chat using simple rules, no magic involved! 🤖",
    "help": "I can chat about simple things. Try saying 'hello', 'joke', 'bye', or ask my name!",
    "who made you": "I was built as Project 1 during a DecodeLabs AI internship. 🛠️",
    "what can you do": "I can greet you, tell a joke, answer simple questions, and say goodbye nicely!",
    "joke": "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
    "thank you": "You're very welcome! 🙌",
    "thanks": "Anytime! 🙌",
    "bye": "Goodbye! Have a great day! 👋",
    "exit": "Goodbye! Have a great day! 👋",
    "quit": "Goodbye! Have a great day! 👋",
}

# Commands that should end the conversation (the "Kill Command")
EXIT_COMMANDS = {"bye", "exit", "quit"}

# Nested-logic follow-up questions: if we don't understand the input,
# ask something clarifying instead of a flat "I don't understand".
CLARIFYING_FOLLOWUPS = [
    "I'm not sure I follow — are you saying hello, asking a question, or trying to leave?",
    "Hmm, I don't have a rule for that yet. Could you try rephrasing, or type 'help'?",
    "That one's outside my rulebook! Want to try 'joke', 'help', or 'bye' instead?",
]


def get_response(user_input: str) -> str:
    """
    Look up a cleaned user input in the knowledge base.
    Falls back to a random clarifying follow-up if there's no match,
    instead of a single flat error message (nested/adaptive fallback).
    """
    if user_input in responses:
        return responses[user_input]
    return random.choice(CLARIFYING_FOLLOWUPS)


def sanitize_input(raw_input: str) -> str:
    """
    Phase 1: Input & Sanitization.
    Normalizes user input so 'Hello', 'HELLO', and ' hello ' all match.
    """
    return raw_input.lower().strip()


def run_chatbot():
    """
    The Heartbeat: an infinite loop that keeps the chatbot 'alive'
    until the user issues an exit command.
    """
    print("RuleBot: Hello! I'm your rule-based chatbot. Type 'bye' to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize_input(raw_input_text)

        reply = get_response(clean_input)
        print(f"RuleBot: {reply}")

        if clean_input in EXIT_COMMANDS:
            break


if __name__ == "__main__":
    run_chatbot()
