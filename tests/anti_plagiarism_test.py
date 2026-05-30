from inference.anti_plagiarism_model_inference import AntiPlagiarismModelInference

AI_SENTENCE = "ИИ помогает людям быстрее находить информацию, учиться и решать разные задачи."
HUMAN_SENTENCE = "Ку, я иду в тц на фудкорт"

def test_anti_plagiarism_model_inference():
    model = AntiPlagiarismModelInference()

    assert model is not None

    ai_score = model.get_label(AI_SENTENCE)
    human_score = model.get_label(HUMAN_SENTENCE)

    assert ai_score != human_score

    assert ai_score == "AI"
    assert human_score == "HUMAN"

    print(f"AI score: {ai_score}, Human score: {human_score}")
    print("All tests passed!")

if __name__ == "__main__":
    test_anti_plagiarism_model_inference()

