from models.anti_plagiarism_previous import load_anti_plagiarism_model
import torch
from data.ai_or_human.dataset_executor import tokenizer, encode_text
from data.ai_or_human.dataset_executor import LABEL_MAP

LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AntiPlagiarismModelInference:
    def __init__(self, model_path, vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=256, output_dim=3):
        self.model = load_anti_plagiarism_model(model_path, vocab_size, embed_dim, hidden_dim, output_dim).to(DEVICE)
        self.model.eval()

    def get_label(self, text):
        inputs = encode_text(text)
        input_tensor = torch.tensor(inputs).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            _, predicted_label = torch.max(output, dim=-1)
            return LABEL_MAP_INV[predicted_label.item()]

if __name__ == "__main__":
    inference_model = AntiPlagiarismModelInference("../anti_plagiarism_model_final.pth")
    text = "I am Jack, your new CEO"
    label = inference_model.get_label(text)
    print(f"Text: {text}")
    print(f"Predicted label for the text: {label}")