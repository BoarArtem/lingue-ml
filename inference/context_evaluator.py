import json

def evaluate_context_with_groq(client, target_phrase: str, user_sentence: str) -> dict:
    prompt = (
        f"Ты - эксперт-лингвист. Проанализируй использование фразы '{target_phrase}' "
        f"в предложении '{user_sentence}'.\n"
        "Верни строго JSON с полями:\n"
        '- "is_used": boolean (есть ли фраза в предложении)\n'
        '- "fits_context": boolean (подходит ли по смыслу)\n'
        '- "sentence_level": string (один из: A1, A2, B1, B2, C1, C2)'
    )
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(completion.choices[0].message.content.strip())

def calculate_fsrs(context_data: dict, error_count: int, expected_level: str) -> int:
    if not context_data.get("is_used") or not context_data.get("fits_context"):
        return 1  # Полное непонимание контекста

    levels = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    actual_level_score = levels.get(context_data.get("sentence_level", "A1"), 1)
    expected_level_score = levels.get(expected_level, 1)

    if error_count == 0:
        if actual_level_score >= expected_level_score:
            return 4  # Идеально
        else:
            return 3  # Без ошибок, но конструкция простовата
    elif error_count <= 2:
        return 3  # Выучил, но есть 1-2 мелкие ошибки
    else:
        return 2  # С трудом