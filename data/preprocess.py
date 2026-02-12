
def vocabulary_expander_preprocess():
    with open("./datasets/words-en.txt", "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
        words = [w for w in words if len(w) >= 3]

        words =  list(set(words))

    return words
