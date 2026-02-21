import ollama

memory = []

system_prompt = """
You are a brutally honest senior tech lead and career mentor for software engineers.
You guide engineers on:
- career growth
- skills to learn
- salary growth
- AI transition
Be practical, structured and honest. No generic advice.
"""

def career_agent(user_input):
    memory.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": system_prompt}] + memory

    response = ollama.chat(
        model="llama3",
        messages=messages
    )

    reply = response['message']['content']
    memory.append({"role": "assistant", "content": reply})

    return reply

print("AI Career Coach Agent Ready (type exit to stop)\n")

while True:
    user = input("You: ")
    if user.lower() == "exit":
        break

    response = career_agent(user)
    print("\n AI Mentor:\n", response, "\n")