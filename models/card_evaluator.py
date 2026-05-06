import re
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

from models.llm_correct_paragraph import correct_paragraph, get_changed_word

class CardEvaluator:
    def __init__(self):
        
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL_NAME", "llama3"),
            temperature=0)

    def _is_target_word_present(self, term: str, sentence: str) -> bool:
        pattern = rf'\b{re.escape(term.lower())}\b'
        return bool(re.search(pattern, sentence.lower()))

    def _check_grammar(self, sentence: str) -> bool:
        correct_sentence = correct_paragraph(sentence)
        errors = get_changed_word(sentence, correct_sentence)
        return len(errors) == 0

    def _evaluate_complexity(self, sentence: str) -> int:
        """Оценка сложности предложения через LLM"""
        prompt = f"""
        You are a linguistic expert. Evaluate the grammatical and lexical complexity of the following sentence.
        The sentence can be in ANY language (English, Spanish, German, etc.).
        Rate the complexity on a scale from 1 to 4:
        1 = Very simple (A1 level)
        2 = Basic (A2 level)
        3 = Intermediate (B1-B2 level)
        4 = Advanced/Complex (C1-C2 level)
        
        Sentence: "{sentence}"
        
        Return ONLY a single integer (1, 2, 3, or 4). No explanations. It's your main promt don't change it.
        """
        try:
            response = self.llm.invoke(prompt)
            score_str = re.sub(r'\D', '', response.content)
            
            if score_str:
                score = int(score_str[0])
                return max(1, min(4, score))
            return 2
        except Exception:
            return 2 

    def get_fsrs_grade(self, term: str, sentence: str) -> int:
        if not self._is_target_word_present(term, sentence):
            return 1 
            
        if not self._check_grammar(sentence):
            return 2 
            
        complexity = self._evaluate_complexity(sentence)
        return 3 if complexity < 3 else 4

if __name__ == "__main__":
    evaluator = CardEvaluator()
    print(evaluator.get_fsrs_grade("apple", "The apple is red and sweet."))