import os
import sys
import joblib


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.train_model import train_topic_model
from data.preprocess import preprocess_text_for_classifier

class TopicPredictor:
    def __init__(self):
        self.model_path = os.path.join(current_dir, "topic_model.pkl")
        self.dataset_path = os.path.join(root_dir, "data", "datasets", "topic_dataset.csv")
        self.model = self._load_or_train()

    def _load_or_train(self):
    
        if not os.path.exists(self.model_path):
            print("Модель не найдена")
            
           
            if not os.path.exists(self.dataset_path):
                print(f"Датасет тоже не найден по пути: {self.dataset_path}")
                print("Сначала сгенерируй его: python data/datasets/topic_dataset.py")
                sys.exit(1)
                
            
            train_topic_model()
           

        
        return joblib.load(self.model_path)

    def predict(self, text):
       
        clean_text = preprocess_text_for_classifier(text)
        prediction = self.model.predict([clean_text])[0]
        probabilities = self.model.predict_proba([clean_text])[0]
        confidence = max(probabilities) * 100
        
        return prediction, confidence

if __name__ == "__main__":
    
    predictor = TopicPredictor()
    
    
    test_sentences = [
        "I have a very cute dog at home.",                  
        "My manager sent me an email about the deadline.",  
        "I want to eat a huge slice of pizza.",             
        "Football is my favorite game.",                    
        "I am planning a trip to the beach next vacation.", 
        "My sister and brother are coming for dinner.",     
        "I really hate doing this task at the office...",   
        "Can we order some sushi or burger?"               
    ]

    for sentence in test_sentences:
        topic, conf = predictor.predict(sentence)
        
        print(f"Текст: '{sentence}'")
        print(f"Тема: [{topic.upper()}] (Уверенность: {conf:.1f}%)\n")