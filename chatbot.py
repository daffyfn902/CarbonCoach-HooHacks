import os
from google import genai
from dotenv import load_dotenv
import model
import app

# 1. Initialize the client
# Ensure your GEMINI_API_KEY is set in your environment variables
load_dotenv()

# 2. Access the variable using os.getenv
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

system_prompt = f"""
You are a sustainability assistant.

User data:
- Transportation: {transport}
- Diet: {diet}
- Energy use: {energy}

Model prediction:
- Estimated carbon footprint: {prediction} metric tons/year

Give personalized, practical ways to reduce this footprint.
Be specific and prioritize biggest impact actions.
"""


def start_chat():
    # 2. Create a chat session with a system instruction
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config={
            "system_instruction": system_prompt
        }
    )

    print("--- Gemini Chat Started (Type 'quit' to exit) ---")

    while True:
        # 3. Get user input
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Gemini: Goodbye!")
            break

        try:
            # 4. Send message and get response
            response = chat.send_message(user_input)
            
            # 5. Print the AI's response
            print(f"Gemini: {response.text}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()