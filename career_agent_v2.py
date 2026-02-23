import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
# -------------------------
# MCP LAYERS
# -------------------------

memory = []
resume_context = ""
knowledge_chunks = []
index = None

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# SYSTEM LAYER
# -------------------------

mentor_mode = "ai"   # default

def get_system_prompt():
    if mentor_mode == "ai":
        return """
You are a brutally honest AI career mentor.
Guide software engineers to transition into AI roles.

Respond in structured format:
1. Current Situation
2. Skill Gaps
3. Exact Roadmap
4. Salary Impact
5. Hard Truth
"""

    elif mentor_mode == "backend":
        return """
You are a senior backend architect and career mentor.
Help engineers grow into senior backend, staff and principal roles.

Focus on:
- system design
- distributed systems
- backend scaling
- promotions
- salary growth

Be practical and direct.
"""

    elif mentor_mode == "interview":
        return """
You are a FAANG-level interview mentor.
Help users crack backend and system design interviews.

Give:
- preparation roadmap
- DSA strategy
- system design guidance
- mock interview style answers
"""

    else:
        return "You are a helpful tech mentor."

# -------------------------
# USER CONTEXT LAYER
# -------------------------

def load_resume(file_path):
    global resume_context

    try:
        if file_path.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            resume_context = text

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                resume_context = f.read()

        print("\n Resume loaded successfully")
        print("FIRST 500 CHARS FROM RESUME:\n")
        print(resume_context[:500])
        print("\n------------------------------\n")

    except Exception as e:
        print("Error loading resume:", e)

# -------------------------
# KNOWLEDGE LAYER (RAG)
# -------------------------

def load_knowledge(file_path):
    global knowledge_chunks, index

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    knowledge_chunks = text.split("\n\n")

    # Create embeddings
    embeddings = embedder.encode(knowledge_chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    print("Knowledge base indexed with FAISS.\n")

def retrieve_context(query, top_k=3):
    if index is None:
        return ""

    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)

    retrieved = [knowledge_chunks[i] for i in indices[0]]
    return "\n".join(retrieved)

# -------------------------
# CONTEXT ORCHESTRATOR
# -------------------------

def build_context(user_input):
    context_messages = [{"role": "system", "content": get_system_prompt()}]

    if resume_context:
        context_messages.append({
            "role": "system",
            "content": f"User Resume:\n{resume_context}"
        })

    retrieved_knowledge = retrieve_context(user_input)
    if retrieved_knowledge:
        context_messages.append({
            "role": "system",
            "content": f"Relevant Expert Knowledge:\n{retrieved_knowledge}"
        })

    return context_messages

# -------------------------
# CAREER AGENT
# -------------------------

def career_agent(user_input):
    memory.append({"role": "user", "content": user_input})

    context_messages = build_context(user_input)
    messages = context_messages + memory

    response = ollama.chat(
        model="llama3",
        messages=messages
    )

    reply = response['message']['content']
    memory.append({"role": "assistant", "content": reply})
    return reply

# -------------------------
# CLI LOOP
# -------------------------

print("AI Mentor MCP + RAG Ready")
print("Commands: upload resume | load knowledge | exit\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    # -------- SWITCH MENTOR MODE --------
    if user.lower().startswith("switch"):
        try:
            mode = user.split(" ")[1]

            if mode in ["ai", "backend", "interview"]:
                mentor_mode = mode
                print(f"\nSwitched to {mode} mentor mode\n")
            else:
                print("\nAvailable modes: ai | backend | interview\n")

        except:
            print("\nUsage: switch ai  OR  switch backend  OR  switch interview\n")

        continue

    # -------- UPLOAD RESUME --------
    if user.lower() == "upload resume":
        path = input("Resume path: ")
        load_resume(path)
        continue

    # -------- LOAD KNOWLEDGE --------
    if user.lower() == "load knowledge":
        path = input("Knowledge file path: ")
        load_knowledge(path)
        continue

    # -------- NORMAL CHAT --------
    response = career_agent(user)
    print("\nAI Mentor:\n", response, "\n")