import os
import re
import asyncio
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from models.llm_correct_paragraph import correct_paragraph, get_changed_word

load_dotenv()


class CardEvaluator:
    def __init__(self):
        
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
        self.llm = ChatOllama(model=model_name, temperature=0)

    async def _count_grammar_errors(self, sentence: str) -> int:
        try:
            ai_sentence = await asyncio.to_thread(correct_paragraph, sentence)
            if not ai_sentence or ai_sentence.strip().lower() == sentence.strip().lower():
                return 0 
                
            incorrect_words, _ = await asyncio.to_thread(get_changed_word, sentence, ai_sentence)
            return len(incorrect_words)
            
        except Exception as e:
            print(f"Ошибка проверки: {e}")
            return 0  

    async def get_fsrs_grade(self, term: str, sentence: str) -> int:
        
        errors_count = await self._count_grammar_errors(sentence)
        system_prompt = f"""Ты - строгий алгоритм оценки.
Твоя задача — проверить использование слова '{term}' в предложении '{sentence}'.
Количество грамматических ошибок в предложении: {errors_count}.

Верни ТОЛЬКО ОДНУ ЦИФРУ(1, 2, 3 или 4) строго по этому ТЗ:

1 - Слова '{term}' вообще нет в предложении или его смысл полностью искажен.
2 - Слово есть, но использовано в совершенно неправильном контексте.
3 - Слово есть, контекст верный, но есть грамматические ошибки или неестественное звучание.
4 - Идеальное предложение: слово в правильном контексте и 0 ошибок.
Никаких пояснений. Только цифра!!!!!!!!!!!!"""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            content = str(response.content).strip()
            match = re.search(r'\b([1-4])\b', content)

            if match:
                grade = int(match.group(1))
                
                if grade == 4 and errors_count > 0:
                    return 3
                    
                return grade
                
            return 2 
            
        except Exception as e:
            print(f"Ошибка вызова Qwen: {e}")
            return 1 