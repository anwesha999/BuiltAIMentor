import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber

# -------------------------
# GLOBAL CONTEXT
# -------------------------

memory = []
resume_context = ""
knowledge_chunks = []
index = None

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# KNOWLEDGE (RAG)
# -------------------------

def load_knowledge(file_path):
    global knowledge_chunks, index

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    knowledge_chunks = text.split("\n\n")
    embeddings = embedder.encode(knowledge_chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    print("Knowledge indexed.\n")


def retrieve_context(query, top_k=3):
    if index is None:
        return ""

    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)

    retrieved = [knowledge_chunks[i] for i in indices[0]]
    return "\n".join(retrieved)


# -------------------------
# RESUME LOADER
# -------------------------

def load_resume(file_path):
    global resume_context

    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    resume_context = text
    print("Resume loaded.\n")


# -------------------------
# BASE AGENT CLASS
# -------------------------

class BaseAgent:
    def __init__(self, role_prompt):
        self.role_prompt = role_prompt

    '''def run(self, user_input):
        knowledge = retrieve_context(user_input)

        messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "system", "content": f"Resume:\n{resume_context}"},
            {"role": "system", "content": f"Relevant Knowledge:\n{knowledge}"},
            {"role": "user", "content": user_input}
        ]

        response = ollama.chat(model="llama3", messages=messages)
        return response['message']['content']
     '''
    def run(self, user_input, shared_context=""):
        # Step 1: Retrieve knowledge (RAG)
        knowledge = retrieve_context(user_input)

        # Step 2: Build messages (MCP-style context)
        messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "system", "content": f"User Resume:\n{resume_context}"},
            {"role": "system", "content": f"Relevant Knowledge:\n{knowledge}"},
        ]

        # Step 3: Add previous agent outputs if available
        if shared_context:
            messages.append({
                "role": "system",
                "content": f"Previous Analysis from other agents:\n{shared_context}"
            })

        # Step 4: Add user input
        messages.append({
            "role": "user",
            "content": user_input
        })

        # Step 5: Call LLM
        response = ollama.chat(
            model="llama3",
            messages=messages
        )

        return response['message']['content']
# -------------------------
# SPECIALIZED AGENTS
# -------------------------

resume_agent = BaseAgent("""
You are a Resume Analysis Agent.
Analyze strengths and weaknesses in the resume.
Return bullet points.
""")

skill_gap_agent = BaseAgent("""
You are a Skill Gap Agent.
Identify missing skills required to reach next career level.
Be specific.
""")

roadmap_agent = BaseAgent("""
You are a Roadmap Agent.
Create a 6 month structured roadmap.
""")

salary_agent = BaseAgent("""
You are a Salary Strategy Agent.
Suggest strategy to increase compensation.
Be practical.
""")

interview_agent = BaseAgent("""
You are an Interview Preparation Agent.
Suggest preparation plan and mock strategy.
""")


# -------------------------
# ORCHESTRATOR
# -------------------------

'''def orchestrator(user_input):

    print("\nRunning Resume Analysis...\n")
    resume_output = resume_agent.run(user_input)

    print("Running Skill Gap Analysis...\n")
    skill_output = skill_gap_agent.run(user_input)

    print("Generating Roadmap...\n")
    roadmap_output = roadmap_agent.run(user_input)

    print("Building Salary Strategy...\n")
    salary_output = salary_agent.run(user_input)

    print("Preparing Interview Plan...\n")
    interview_output = interview_agent.run(user_input)

    final_response = f"""
================ CAREER INTELLIGENCE REPORT ================

RESUME ANALYSIS:
{resume_output}

SKILL GAPS:
{skill_output}

ROADMAP:
{roadmap_output}

SALARY STRATEGY:
{salary_output}

INTERVIEW PLAN:
{interview_output}
"""

    return final_response
'''

def orchestrator(user_input):

    shared_context = ""

    print("\nRunning Resume Analysis...\n")
    resume_output = resume_agent.run(user_input)
    shared_context += f"\nRESUME:\n{resume_output}\n"

    print("Running Skill Gap Analysis...\n")
    skill_output = skill_gap_agent.run(user_input, shared_context)
    shared_context += f"\nSKILL GAPS:\n{skill_output}\n"

    print("Generating Roadmap...\n")
    roadmap_output = roadmap_agent.run(user_input, shared_context)
    shared_context += f"\nROADMAP:\n{roadmap_output}\n"

    print("Building Salary Strategy...\n")
    salary_output = salary_agent.run(user_input, shared_context)
    shared_context += f"\nSALARY:\n{salary_output}\n"

    print("Preparing Interview Plan...\n")
    interview_output = interview_agent.run(user_input, shared_context)

    final_response = f"""
================ CAREER INTELLIGENCE REPORT ================

RESUME ANALYSIS:
{resume_output}

SKILL GAPS:
{skill_output}

ROADMAP:
{roadmap_output}

SALARY STRATEGY:
{salary_output}

INTERVIEW PLAN:
{interview_output}
"""

    return final_response

# -------------------------
# CLI
# -------------------------

print("Career Agent V3 (Multi-Agent) Ready\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    print(orchestrator(user))