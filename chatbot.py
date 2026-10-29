
import csv
import os
from datetime import datetime

import ollama

# Project folders
DATASET = "dataset"
os.makedirs(DATASET, exist_ok=True)

CHAT_HISTORY_FILE = os.path.join(DATASET, "chat_history.csv")
RESPONSES_FILE = os.path.join(DATASET, "responses.csv")
LOG_FILE = os.path.join(DATASET, "chatbot_logs.txt")

MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = """
You are an Education Assistant chatbot.
Answer questions about learning, school, college,
exams, study methods, programming, and career guidance.
Use simple English and give clear examples.
If a question is outside education, politely explain
that your main purpose is to help with education.
"""

# Create CSV files with headers if they do not exist
def create_csv_file(filename, headers):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)


create_csv_file(
    CHAT_HISTORY_FILE,
    ["timestamp", "user_question", "chatbot_response"]
)

create_csv_file(
    RESPONSES_FILE,
    ["timestamp", "user_question", "chatbot_response"]
)


# Save one conversation record
def save_conversation(question, answer):
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    record = [timestamp, question, answer]

    for filename in [CHAT_HISTORY_FILE, RESPONSES_FILE]:
        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(record)

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}]\n")
        file.write(f"User: {question}\n")
        file.write(f"Chatbot: {answer}\n\n")


# Main chatbot function
def main():
    print("=" * 50)
    print("       EDUCATION AI CHATBOT")
    print("       Powered by Ollama")
    print("=" * 50)
    print("Ask me education-related questions.")
    print("Type 'exit' to stop the chatbot.\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Chatbot: Thank you! Goodbye.")
            break

        if not question:
            print("Chatbot: Please enter a question.")
            continue

        messages.append({
            "role": "user",
            "content": question
        })

        try:
            result = ollama.chat(
                model=MODEL_NAME,
                messages=messages
            )

            answer = result["message"]["content"]

            print("\nChatbot:", answer, "\n")

            messages.append({
                "role": "assistant",
                "content": answer
            })

            save_conversation(question, answer)

        except Exception as error:
            print("Error:", error)
            print("Check whether Ollama is running and")
            print("whether the model llama3.2 is installed.")

            # Remove the unanswered question from context
            messages.pop()


if __name__ == "__main__":
    main()
