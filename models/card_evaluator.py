import re, spacy
from dotenv import load_dotenv 

load_dotenv() 

from models.llm_correct_paragraph import correct_paragraph, get_changed_word

nlp = spacy.load('en_core_web_sm')

class CardEvaluator:
    def __init__(self):
        pass

    def _is_target_word_present(self, term: str, sentence: str) -> bool:
        pattern = rf'\b{re.escape(term.lower())}\b'
        return bool(re.search(pattern, sentence.lower()))

    def _check_grammar(self, sentence: str) -> bool:
        correct_sentence = correct_paragraph(sentence)
        errors = get_changed_word(sentence)

        return len(errors) == 0

    def _evaulate_complexity(self, sentence: str) -> int:
        '''Оценка сложности предложения с помощью spacy'''

        doc = nlp(sentence)

        if len(doc) < 4:
            return 1
    
        complex_class = [token for token in doc if token.pos_ == "SCONJ"]
        max_tree_depth = max([len(list(token.ancestors)) for token in doc], default= 0)
    
        if len(complex_class) > 0 or max_tree_depth >= 3 or len(doc) >= 8:
            return 2 
            
        return 1 

    def get_fsrs_grade(self, term: str, sentence: str) -> int:
        """Объединяет проверки и выдает оценку 1-4 для FSRS"""
        if not self._is_target_word_present(term, sentence):
            return 1

        is_grammar_correct = self._check_grammar(sentence)
        complexity_level = self._evaluate_complexity(sentence)

        if not is_grammar_correct:
            return 2

       
        if is_grammar_correct and complexity_level == 1:
            return 3

    
        if is_grammar_correct and complexity_level >= 2:
            return 4

        return 1
    


