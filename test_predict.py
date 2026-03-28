# test_predict.py
# Проверка работы классификатора Lingue

from models.words_classifier import predict  

def main():
    print("Тестируем модель Lingue")
    while True:
        text = input("Введи слово или тему (для выхода '0'): ").strip()
        if text.lower() == '0':
            break
        try:
            result = predict(text)  
            print(f"Слово '{text}' относится к теме: {result}\n")
        except Exception as e:
            print(f"Ошибка при предсказании: {e}\n")

if __name__ == "__main__":
    main()