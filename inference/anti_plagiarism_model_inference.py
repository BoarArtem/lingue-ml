from models.anti_plagiarism import load_anti_plagiarism_model
import torch
from data.ai_or_human.dataset_executor import tokenizer, encode_text, tokenize_text
from data.ai_or_human.dataset_executor import LABEL_MAP

class AntiPlagiarismModelInference:
    def __init__(self, vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=256, output_dim=3):
        self.model = load_anti_plagiarism_model("anti_plagiarism_model.pth", vocab_size, embed_dim, hidden_dim, output_dim)
        self.model.eval()

    def get_label(self, text):
        inputs = tokenize_text(text)
        inputs = encode_text(inputs)
        input_tensor = torch.tensor(inputs).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_label = torch.max(output, dim=-1)
            return LABEL_MAP[predicted_label.item()]
