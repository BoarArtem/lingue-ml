import re
import os
import asyncio
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

from models.llm_correct_paragraph import correct_paragraph, get_changed_word

class CardEvaluator:
    def __init__(self):
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
        self.llm = ChatOllama(
            model=model_name,
            temperature=0  
        )

    async def _is_target_word_present(self, term: str, sentence: str) -> bool:
        """Проверка наличия слова с учетом морфологии (любая форма)."""
        if not term or not sentence:
            return False
            
        
        pattern = rf'\b{re.escape(term.lower())}\b'
        if bool(re.search(pattern, sentence.lower())):
            return True
            
        
        messages = [
            SystemMessage(content="You are a strict linguistic morphological analyzer. Check if the provided sentence contains the given base word in ANY of its grammatical forms (declensions, conjugations, plurals, past tense, etc.). Respond ONLY with 'YES' or 'NO'."),
            HumanMessage(content=f"Base word: '{term}'\nSentence: '{sentence}'")
        ]
        try:
            response = await self.llm.ainvoke(messages)
            return "yes" in str(response.content).lower()
        except Exception as e:
            print(f"Ошибка проверки морфологии: {e}")
            return False

    async def _check_grammar(self, sentence: str) -> bool:
        """Проверка грамматики"""
        
        try:
            ai_sentence = await asyncio.to_thread(correct_paragraph, sentence)
            
            
            if not ai_sentence or ai_sentence.strip().lower() == sentence.strip().lower():
                return True 
                
            incorrect_words, correct_words = await asyncio.to_thread(get_changed_word, sentence, ai_sentence)
            
           
            return len(incorrect_words) == 0
        except Exception as e:
            print(f"Ошибка проверки грамматики: {e}")
            
            return True 

    async def _evaluate_complexity(self, sentence: str) -> int:
        """Мультиязычная оценка сложности предложения через Qwen"""
        messages = [
            SystemMessage(content="""You are a strict linguistic expert. 
Evaluate the grammatical and lexical complexity of the user's sentence.
Return ONLY a single digit: 1, 2, 3, or 4.
1 = Very simple (A1 level)
2 = Basic (A2 level)
3 = Intermediate (B1-B2 level)
4 = Advanced/Complex (C1-C2 level)"""),
            HumanMessage(content=f'Sentence: "{sentence}"')
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = str(response.content).strip()
            
            match = re.search(r'\b([1-4])\b', content)
            if match:
                return int(match.group(1))
            return 2 
            
        except Exception as e:
            print(f"Ошибка инференса Qwen: {e}")
            return 2

    async def get_fsrs_grade(self, term: str, sentence: str) -> int:
        """Главный метод для получения оценки"""
        
        has_word = await self._is_target_word_present(term, sentence)
        if not has_word:
            return 1
            
        
        is_correct = await self._check_grammar(sentence)
        if not is_correct:
            return 2
            
       
        complexity = await self._evaluate_complexity(sentence)
        
        return 3 if complexity < 3 else 4