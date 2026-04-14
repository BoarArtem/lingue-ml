import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import re

load_dotenv()
client = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"), temperature=0)


def correct_paragraph(user_sentence, iterations=2):
    """
    Returns: (user_fixed, ai_fixed)
    - user_fixed: original tokens (some may be <ERR> if AI couldn't align)
    - ai_fixed: corrected tokens (with <ERR> for unfixable tokens)
    """

    best_response = None

    for _ in range(iterations):
        response = client.invoke([
            SystemMessage(content="""
You are a grammar correction system. You fix spelling, inflection, and grammar errors while preserving the exact meaning and token structure.

OUTPUT FORMAT (single line, no explanation):
<user_tokens> | <corrected_tokens>

CORE RULES:

1. TOKENIZATION:
   - Split by spaces
   - Punctuation MUST be separate tokens: "hello." → "hello ."
   - Token count in output MUST equal token count in input

2. LEFT SIDE (user_tokens):
   - EXACT copy of input tokens
   - NEVER modify, NEVER add <ERR>
   - Byte-identical to input

3. RIGHT SIDE (corrected_tokens):
   - Same token count as left side
   - Fix spelling: "youu" → "you", "ваду" → "воду"
   - Fix grammar: "go" → "goes", "школа" → "школу"
   - Use <ERR> ONLY when token is unfixable (stranded preposition, nonsense word)
   - NEVER invent new words
   - NEVER change meaning

4. WHEN TO USE <ERR> (right side only):
   ✓ Stranded preposition with no object: "под манго" → "под" is <ERR>
   ✓ Unrecognizable garbage: "xzqwf" → <ERR>
   ✗ NEVER for simple typos (fix them!)
   ✗ NEVER for wrong tense (fix it!)

5. LANGUAGE:
   - Output MUST be same language as input
   - Russian → Russian, English → English
   - NEVER translate

6. MEANING PRESERVATION:
   - Content words (nouns, verbs, adjectives) are SACRED
   - Only fix spelling/grammar, never swap synonyms
   - "кушать" stays "кушать", NOT "есть"
   - "манго" stays "манго", NOT some other word

EXAMPLES:

Input: I lave youu
Output: I lave youu | I love you

Input: Я люблю кушать под манго
Output: Я люблю кушать под манго | Я люблю кушать <ERR> манго

Input: she go school every day .
Output: she go school every day . | she goes school every day .

Input: Я купил ваду пойду в школа
Output: Я купил ваду пойду в школа | Я купил воду пойду в школу

Input: I want eat at dog .
Output: I want eat at dog . | I want eat <ERR> dog .

VALIDATION BEFORE OUTPUT:
1. Count tokens: left == right == input
2. Left side is byte-identical to input
3. No invented words on right side
4. <ERR> only on right side, only for unfixable tokens
"""),
            HumanMessage(content=user_sentence),
        ])

        best_response = response.content.strip()

        # Validate response format
        if ' | ' not in best_response:
            continue

        parts = best_response.split(' | ')
        if len(parts) != 2:
            continue

        user_part, ai_part = parts

        # Validate token counts
        input_tokens = user_sentence.split()
        user_tokens = user_part.split()
        ai_tokens = ai_part.split()

        if len(user_tokens) == len(ai_tokens) == len(input_tokens):
            break

    if best_response is None or ' | ' not in best_response:
        # Fallback: return original
        return user_sentence, user_sentence

    user_fixed, ai_fixed = best_response.split(' | ', 1)
    return user_fixed.strip(), ai_fixed.strip()


def tokenize(text):
    """Tokenize text into words and punctuation"""
    return re.findall(r"\w+|[^\w\s]", text)


def word_pair(user_fixed: str, ai_fixed: str):
    """
    Create word pairs for comparison
    Returns: [(user_token, ai_token), ...]
    """
    user_tokens = user_fixed.split()
    ai_tokens = ai_fixed.split()

    # Pad shorter list if needed
    max_len = max(len(user_tokens), len(ai_tokens))
    user_tokens += [''] * (max_len - len(user_tokens))
    ai_tokens += [''] * (max_len - len(ai_tokens))

    return list(zip(user_tokens, ai_tokens))


def get_corrections(user_fixed: str, ai_fixed: str):
    """
    Extract only the changed tokens
    Returns: [(original, corrected), ...]
    """
    corrections = []

    for user_token, ai_token in word_pair(user_fixed, ai_fixed):
        if user_token != ai_token and ai_token not in ['', '<ERR>']:
            corrections.append((user_token, ai_token))

    return corrections


def get_errors(user_fixed: str, ai_fixed: str):
    """
    Extract tokens marked as <ERR>
    Returns: [original_token, ...]
    """
    errors = []

    for user_token, ai_token in word_pair(user_fixed, ai_fixed):
        if ai_token == '<ERR>' and user_token:
            errors.append(user_token)

    return errors


if __name__ == "__main__":

    test_sentences = [
        "Я люблю кушать под манго",
        "I lave youu",
        "she go school every day",
        "Я купил ваду пойду в школа",
        "I want eat at dog",
        "Я кушаю яблоко вчора",
    ]

    for sentence in test_sentences:
        print(f"\n{'=' * 60}")
        print(f"INPUT: {sentence}")

        user_fixed, ai_fixed = correct_paragraph(sentence)

        print(f"USER:  {user_fixed}")
        print(f"AI:    {ai_fixed}")

        pairs = word_pair(user_fixed, ai_fixed)
        print(f"PAIRS: {pairs}")

        corrections = get_corrections(user_fixed, ai_fixed)
        if corrections:
            print(f"FIXES: {corrections}")

        errors = get_errors(user_fixed, ai_fixed)
        if errors:
            print(f"ERRORS: {errors}")