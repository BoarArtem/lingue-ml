import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import difflib
import re

load_dotenv()
client = ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=0)

def correct_paragraph(user_sentence, iterations=2):
    responses = []
    for _ in range(iterations):
        response = client.invoke([
            SystemMessage(content="""
                You are a text correction tool.

                Output EXACTLY one single line. No newlines. No commentary. No explanation.
                
                Line format: <user_fixed> | <ai_fixed>
                
                Definitions:
                - user_fixed: original user sentence where you ONLY insert "0" tokens for missing words.
                - ai_fixed: fully corrected sentence.
                
                STRICT RULES:
                1. Tokenization:
                - Split by spaces.
                - Every punctuation mark (.,!? etc.) MUST be a separate token with spaces around it.
                
                2. Core principle (MOST IMPORTANT):
                - First, determine the corrected sentence (ai_fixed).
                - Then compare lengths:
                • If lengths are EQUAL → DO NOT insert "0" at all.
                • If ai_fixed is LONGER → insert "0" in user_fixed at missing positions.
                - NEVER insert "0" if it is possible to align tokens by position.
                
                3. Alignment:
                - user_fixed and ai_fixed MUST have EXACTLY the same number of tokens.
                - You may ONLY insert "0" tokens into user_fixed.
                - NEVER modify, delete, reorder, or fix original user words in user_fixed.
                
                4. Replacement vs insertion:
                - If a word is incorrect (e.g., "lave" → "love") → this is a REPLACEMENT.
                - DO NOT insert "0" for replacements.
                - Keep the original word in user_fixed, fix it ONLY in ai_fixed at the same position.
                
                5. Zero insertion:
                - Insert "0" ONLY when a word is completely missing (ai_fixed has an extra token).
                - "0" can ONLY be inserted between tokens or at boundaries.
                - Number of "0" tokens MUST exactly equal missing words.
                - DO NOT insert extra "0"s.
                
                6. NO FIXING USER TEXT:
                - DO NOT correct spelling, grammar, punctuation, or wording in user_fixed.
                - DO NOT split or merge tokens.
                - user_fixed = original tokens + optional "0" only.
                
                7. Context preservation:
                - NEVER change or remove any user tokens for context reasons.
                - Keep user text exactly as provided; context must be preserved.
                - Only insert "0" if a word is missing according to ai_fixed.
                
                8. ai_fixed:
                - Must be grammatically correct.
                - Must align 1-to-1 with user_fixed.
                
                9. Validation (MANDATORY before output):
                - len(user_fixed_tokens) == len(ai_fixed_tokens)
                - If lengths were originally equal → there must be ZERO "0" tokens.
                - Each "0" corresponds to exactly one extra token in ai_fixed.
                - No extra or missing tokens.
                
                Examples:
                
                Input: I like eat cheese pizza , my dog also  
                Output: I like 0 eat cheese pizza , my dog also | I like to eat cheese pizza , my dog also  
                
                Input: I lave youu  
                Output: I lave youu | I love you  
            """
            ),
            HumanMessage(content=user_sentence),
        ])

        responses.append(response.content)




    user_fixed, ai_fixed = responses[-1].split(' | ')

    user_fixed_with_correct_zeros = []

    # if len(user_fixed.split()) != len(ai_fixed.split()):
    #     for user_token in user_fixed.split():
    #         if user_token.startswith("0"):
    #             user_fixed = user_fixed.split().remove(user_token)
    #         else:
    #             user_fixed_with_correct_zeros.append(user_token)


    if "0" in user_fixed.split():
        for user_token, ai_token in zip(user_fixed.split(), ai_fixed.split()):
            if user_token.startswith("0"):
                if len(list(ai_token)) == 0:
                    continue
                else:
                     
            else:
                user_fixed_with_correct_zeros.append(user_token)


    user_fixed = " ".join(user_fixed_with_correct_zeros)



    return user_fixed, ai_fixed

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

def get_changed_word(user_sentence, corrected_sentence):
    correct_changes = []
    incorrect_changes = []

    user_words = tokenize(user_sentence)
    corrected_words = tokenize(corrected_sentence)

    diff = difflib.ndiff(user_words, corrected_words)

    for d in diff:
        token = d[2:]

        if d.startswith("+ "):
            correct_changes.append(token)
        elif d.startswith("- "):
            # if re.fullmatch(r"[^\w\s]", token):
            incorrect_changes.append(token)

    return correct_changes, incorrect_changes



if __name__ == "__main__":
    # user_sentence = "Я! завтра сьел кашу"
    # correct_sentence = correct_paragraph(user_sentence)
    #
    # correct_words, incorrect_words = get_changed_word(user_sentence, correct_sentence)
    #
    # ai_checking = {
    #     "correction": correct_sentence,
    #     "corrected_word": correct_words,
    #     "incorrected_word": incorrect_words
    # }

    user_sentence = "Я купил ваду пожтому пойду в школа"

    user_fixed, ai_fixed = correct_paragraph(user_sentence)

    print(user_fixed.split())
    print(ai_fixed.split())

    print(len(user_fixed.split()))
    print(len(ai_fixed.split()))