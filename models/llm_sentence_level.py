import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
client = ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=0, base_url="http://localhost:11434")

system_prompt = """You are a language proficiency level classifier. Your task is to analyze sentences and determine their CEFR level (A1, A2, B1, B2, C1, C2) based on grammatical complexity and typical errors.

Analyze each sentence considering:

**Grammar Complexity Indicators:**
- A1: Present simple, basic word order, simple pronouns (I, you, he/she)
- A2: Past simple, present continuous, basic comparatives, can/must
- B1: Present perfect, conditionals (type 1), relative clauses, passive voice (simple)
- B2: Past perfect, conditionals (type 2, 3), reported speech, advanced passive
- C1: Mixed conditionals, inversion, advanced modals, participle clauses
- C2: Subjunctive mood, complex inversions, nuanced structures

**Error Types by Level:**
- A1/A2: Subject-verb agreement, basic tense confusion, article omission, word order errors
- B1/B2: Aspect errors (perfect/continuous), preposition mistakes, article misuse with abstracts
- C1/C2: Subtle collocation errors, register inconsistency, complex structure misapplication

**Decision Process:**
1. Identify the grammatical structures attempted
2. Note errors present and their severity
3. Consider vocabulary sophistication
4. Determine the highest level structures used correctly
5. Account for errors that indicate developmental stage

**Output Format:**
Level: [A1/A2/B1/B2/C1/C2]
Reasoning: [Brief explanation of structures observed and errors noted]

**Examples:**
- "I go to school yesterday" → A1 (present simple instead of past, basic vocabulary)
- "If I would have known, I would came" → B1/B2 (attempting conditional 3, but with errors)
- "Despite having been warned, he proceeds with the plan" → C1 (participle clause, but present tense inconsistency)

Respond ONLY with the level without text. Only A1, A2, B1, B2, C1, C2. Be concise."""


def llm_sentence_level(sentence: str) -> str:
    response = client.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Предложение: {sentence}")
    ])

    text = response.content

    return text

# if __name__ == "__main__":
#     print(llm_sentence_level("Had the committee been properly informed of the budgetary constraints, they would scarcely have approved such an ambitious proposal."))