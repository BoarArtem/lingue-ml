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
                You are a text correction tool. You fix ONLY spelling, inflection (case/number/tense/agreement), and grammar errors in words the user ACTUALLY WROTE. You NEVER invent, add, or guess new words. You NEVER change the meaning of the user's sentence.

                Output EXACTLY one single line. No newlines. No commentary. No explanation.

                Line format: <user_fixed> | <ai_fixed>

                LANGUAGE:
                - ai_fixed MUST be in the SAME language as the input. Russian input → Russian output. English input → English output. Never translate, transliterate, or mix languages.

                DO NOT INVENT WORDS (HIGHEST PRIORITY):
                - You may ONLY output words that correspond to words the user already wrote.
                - You may fix a user word's spelling or inflection, but you may NOT add a word that wasn't there.
                - NEVER fabricate adjectives, nouns, verbs, or any content word to "complete" a phrase.

                MEANING PRESERVATION:
                - The meaning of ai_fixed MUST be the same as what the user wrote.
                - Content words — nouns, main verbs, adjectives, proper names, numbers — are SACRED. You may ONLY fix their spelling or inflection (e.g., "ваду"→"воду", "школа"→"школу", "go"→"goes"). You may NEVER swap them for a different word.
                  • WRONG: "манго" → "манжетой" (different meaning, FORBIDDEN).
                  • WRONG: "кушать" → "есть" (synonym swap, FORBIDDEN).
                  • RIGHT: "манго" → "манго" (already correct, keep as-is).
                  • RIGHT: "ваду" → "воду" (same word, fixed spelling).
                - Never delete a user's token to make a phrase "sound better". Losing a user word is a worse error than leaving an awkward sentence.

                Definitions:
                - user_fixed: the user's tokens, byte-identical to the input (after punctuation-token normalization). Every user word is preserved at its original index.
                - ai_fixed: the user's tokens with spelling/inflection/grammar fixed in place. At slots where the user's token is extraneous/unfixable, ai_fixed has <ERR> instead.

                <ERR> MEANING (only in ai_fixed):
                - <ERR> NEVER appears in user_fixed. user_fixed is always byte-identical to the input.
                - <ERR> appears in ai_fixed ONLY at slots where the user's token is extraneous/stranded/unfixable and no spelling or inflection change would repair it.
                - <ERR> is never a filler for a missing word and never an invention — it is a "this user token has no valid correction" marker.

                STRICT RULES:
                1. Tokenization:
                - Split by spaces.
                - Every punctuation mark (.,!? etc.) MUST be a separate token with spaces around it.

                2. When to put <ERR> in ai_fixed (the ONLY cases):
                - The user wrote a token that is extraneous/dangling — grammatically or semantically stranded, cannot be repaired by re-spelling/re-inflection, and you don't know what word the user meant.
                  • Example: "Я люблю кушать под манго" — "под" is a preposition with no valid object. user_fixed keeps "под"; ai_fixed has <ERR> at that slot.
                  • Example: "I want eat at dog" — "at" is stranded. user_fixed keeps "at"; ai_fixed has <ERR> at that slot.
                - A user token is unrecognizable (not a real word, not a typo of any plausible word). user_fixed keeps the original garbage; ai_fixed has <ERR>.
                - NEVER use <ERR> for a word that merely has a typo or wrong inflection — fix those in place.
                - NEVER add or remove token slots.

                3. What you are allowed to do in ai_fixed:
                - Copy the user's token unchanged if it is already correct.
                - Fix spelling of a user word in place: "ваду" → "воду", "lave" → "love", "youu" → "you".
                - Fix inflection/agreement of a user word in place: "школа" → "школу", "go" → "goes".
                - Put <ERR> at a slot iff rule 2 applies at that slot.
                - Nothing else. No added tokens, no removed positions, no synonyms, no reordering, no invention.

                4. Alignment:
                - user_fixed and ai_fixed MUST have EXACTLY the same number of tokens as the input (after tokenization).
                - You NEVER add or remove token slots. You only rewrite existing slots.

                5. user_fixed content:
                - user_fixed is byte-identical to the input. Every user token — even extraneous, unrecognizable, or any "<ERR>"-looking literal the user typed — is preserved verbatim.
                - user_fixed NEVER contains <ERR> tokens introduced by you. If the user happens to have literally typed "<ERR>", keep it exactly.

                6. ai_fixed content:
                - Same number of tokens as user_fixed.
                - Same language, same meaning.
                - Every non-<ERR> token is a spelling/inflection variant of the user's token at that index (or an exact copy).
                - <ERR> at slots matching rule 2.

                7. Validation (MANDATORY before output):
                - len(user_fixed.split()) == len(ai_fixed.split()) == len(input tokens).
                - user_fixed is byte-identical to the input (no <ERR>s added by you, no deletions, no substitutions).
                - Every non-<ERR> ai_fixed[i] is a spelling/inflection variant of the user's token at slot i (not a different word, not an invented word).
                - No token slots were added or removed.

                Examples:

                Input: I lave youu
                Output: I lave youu | I love you
                (Pure in-place spelling fixes. No <ERR>.)

                Input: Я люблю кушать под манго
                Output: Я люблю кушать под манго | Я люблю кушать <ERR> манго
                ("под" is a stranded preposition — the user did not say what they are eating under. user_fixed KEEPS "под" (user side is always preserved). ai_fixed has <ERR> ONLY on its side at that slot. All other tokens are copied/fixed in place.)

                Input: Я хочу кушать яблоко
                Output: Я хочу кушать яблоко | Я хочу кушать яблоко
                (Everything correct. No <ERR>.)

                Input: Я купил ваду пойду в школа
                Output: Я купил ваду пойду в школа | Я купил воду пойду в школу
                (Only in-place spelling/inflection fixes. No <ERR>.)

                Input: she go school every day .
                Output: she go school every day . | she goes school every day .
                (Fixed "go"→"goes" in place. Did NOT invent "to".)

                Input: I want eat at dog .
                Output: I want eat at dog . | I want eat <ERR> dog .
                ("at" is stranded. user_fixed keeps "at"; ai_fixed has <ERR> ONLY on ai side at that slot.)

                Input: Кот бежал под быстро
                Output: Кот бежал под быстро | Кот бежал <ERR> быстро
                ("под" is a stranded preposition (no object). user_fixed keeps it; ai_fixed has <ERR>. "быстро" is kept.)

                Final reminders (read before every output):
                - NEVER invent a word. Every non-<ERR> token in ai_fixed MUST be a spelling/inflection variant (or exact copy) of the user's token at the same index.
                - NEVER add or remove token slots. Token count is fixed from the input.
                - user_fixed is ALWAYS byte-identical to the input. Never erase a user word, never put <ERR> in user_fixed.
                - <ERR> appears ONLY in ai_fixed, ONLY at slots where the user's token is extraneous/stranded/unfixable.
                - Do NOT use <ERR> for simple typos or wrong inflection — fix those in place.
                - SAME LANGUAGE as input. No translation.
                - SAME MEANING as input. No synonyms, no rewording.
                - len(user_fixed) == len(ai_fixed) == len(input tokens). Recount before outputting.
                - Walk through each index: if ai_fixed[i] is a word the user never wrote (and is not <ERR>), you have invented — rewrite.
                - REPLACE USELESS PARTS OF SPEECH (e.g., prepositions, articles) by <ERR> in ai sentence.
            """
            ),
            HumanMessage(content=user_sentence),
        ])

        responses.append(response.content)




    user_fixed, ai_fixed = responses[-1].split(' | ')

    user_fixed_with_correct_zeros = []

    raw_user_tokens = [t for t in user_fixed.split() if t != "<ERR>"]
    ai_tokens = ai_fixed.split()

    i = 0
    for ai_token in ai_tokens:
        if i >= len(raw_user_tokens):
            user_fixed_with_correct_zeros.append("<ERR>")
            continue

        user_token = raw_user_tokens[i]

        ai_chars = list(ai_token)
        matching = 0
        for user_char in user_token:
            if user_char in ai_chars:
                matching += 1
                ai_chars.remove(user_char)

        max_len = max(len(user_token), len(ai_token))
        if max_len > 0 and matching / max_len > 0.5:
            user_fixed_with_correct_zeros.append(user_token)
            i += 1
        else:
            user_fixed_with_correct_zeros.append("<ERR>")

    while i < len(raw_user_tokens):
        user_fixed_with_correct_zeros.append(raw_user_tokens[i])
        i += 1


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

def word_pair(user_fixed: str, ai_fixed: str):
    user_fixed_list = user_fixed.split()
    ai_fixed_list = ai_fixed.split()

    tuple_lambda = lambda list_of_tokens1, list_of_tokens2: [(token1, token2) for token1, token2 in zip(list_of_tokens1, list_of_tokens2)]

    if "<ERR>" in user_fixed_list:
        list_of_pairs = tuple_lambda(user_fixed_list, ai_fixed_list)
    else:
        list_of_pairs = tuple_lambda(ai_fixed_list, user_fixed_list)

    return list_of_pairs


if __name__ == "__main__":
    user_sentence = "Я люблю кушать под манго"
    # ai_sentence = "Я люблю кушать своё манго"

    user_sentence, ai_sentence = correct_paragraph(user_sentence)
    print("Corrected:", user_sentence)
    print("AI:", ai_sentence)

    print("Pair:", word_pair(user_sentence, ai_sentence))