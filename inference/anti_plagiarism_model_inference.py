from models.anti_plagiarism import get_plagiarism_score, convert_text_to_label

class AntiPlagiarismModelInference:
    def get_label(self, text):
        return get_plagiarism_score(text)

    def get_index_from_label(self, score):
        return convert_text_to_label(score)

if __name__ == "__main__":
    model = AntiPlagiarismModelInference()
    text = "I am Jack, your new CEO"
    label = model.get_label(text)
    print(f"Text: {text}")
    print(f"Predicted label for the text: {label}")