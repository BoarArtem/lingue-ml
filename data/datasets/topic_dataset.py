'''Генератор датасета'''
import random
import pandas as pd
import os

random.seed(42)


categories = {
    "animals": ["cat", "dog", "lion", "tiger", "elephant", "bird", "rabbit", "horse", "wolf", "fox", "bear", "deer"],
    "family": ["mother", "father", "parents", "brother", "sister", "aunt", "uncle", "cousin", "grandmother", "grandfather", "wife", "husband"],
    "relationships": ["love", "partner", "marriage", "couple", "dating", "boyfriend", "girlfriend", "wedding", "relationship", "romance", "kiss", "hug"],
    "friends": ["friend", "buddy", "mate", "hangout", "friendship", "companion", "peer", "bestie", "pal", "neighbor", "group", "gang"],
    "emotions": ["happy", "sad", "angry", "joy", "fear", "surprise", "excitement", "stress", "anxiety", "relief", "cry", "smile"],
    "health": ["doctor", "hospital", "medicine", "disease", "health", "pain", "workout", "pill", "clinic", "treatment", "sick", "nurse"],
    "education": ["school", "teacher", "student", "university", "exam", "book", "lesson", "class", "homework", "degree", "college", "study"],
    "work": ["meeting", "job", "project", "task", "office", "deadline", "email", "manager", "team", "report", "boss", "salary"],
    "technology": ["computer", "phone", "internet", "software", "app", "screen", "network", "code", "data", "device", "laptop", "robot"],
    "entertainment": ["movie", "music", "game", "concert", "show", "theatre", "art", "festival", "party", "dance", "cinema", "song"],
    "sports": ["football", "tennis", "basketball", "running", "gym", "training", "match", "stadium", "fitness", "swimming", "player", "coach"],
    "food": ["pizza", "burger", "apple", "pasta", "coffee", "salad", "chicken", "bread", "soup", "rice", "meat", "cake"],
    "shopping": ["store", "price", "buy", "sell", "discount", "mall", "shop", "money", "clothes", "cart", "market", "pay"],
    "home": ["house", "apartment", "room", "furniture", "bed", "kitchen", "bathroom", "door", "window", "roof", "garden", "wall"],
    "travel": ["trip", "journey", "flight", "vacation", "travel", "tour", "hotel", "ticket", "airport", "beach", "passport", "guide"],
    "transport": ["car", "bus", "train", "subway", "bike", "taxi", "road", "station", "airplane", "boat", "ship", "drive"],
    "finance": ["money", "bank", "cash", "credit", "card", "investment", "salary", "tax", "budget", "economy", "wallet", "coin"],
    "time": ["day", "night", "hour", "minute", "future", "past", "tomorrow", "yesterday", "clock", "calendar", "month", "year"],
    "nature": ["tree", "forest", "river", "mountain", "sky", "ocean", "sun", "rain", "wind", "flower", "grass", "snow"],
    "other": ["thing", "stuff", "object", "piece", "item", "something", "anything", "detail", "part", "matter", "tool", "box"]
}

templates_pool = [
    "I like {}", "I love {}", "I have a {}", "This is my {}", "I see the {}",
    "I really like {}", "I often use {} in my life", "I saw {} yesterday", "I want to try {} soon", "{} is very important to me",
    "Yesterday I was thinking about {} and its importance", "In my opinion, {} plays a big role in our life", "I have always been interested in {}", "Sometimes I cannot imagine my life without {}", "People often underestimate how important {} is",
    "Do you like {}?", "Have you ever tried {}?", "Why do people love {}?", "What do you think about {}?", "Where can I find {}?",
    "Even though I was busy, I still thought about {}", "While working, I realized how important {} is", "After a long day, I decided to focus on {}", "Before going to sleep, I remembered {}"
]

prefixes = ["", "", "Honestly,", "Well,", "Actually,", "You know,", "Basically,", "To be honest,"]
suffixes = ["", "", "these days.", "for me.", "in my opinion.", "right now.", "as usual.", "every day."]

def build_sentence(word):
    template = random.choice(templates_pool)
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    sentence = template.format(word)
    full = f"{prefix} {sentence} {suffix}".strip()
    return " ".join(full.split())

def generate_dataset(samples_per_class=500):
    texts, labels = [], []
    for label, words in categories.items():
        used = set()
        count = 0
        max_attempts = samples_per_class * 20
        attempts = 0

        while count < samples_per_class and attempts < max_attempts:
            word = random.choice(words)
            sentence = build_sentence(word)

            if sentence not in used:
                used.add(sentence)
                texts.append(sentence)
                labels.append(label)
                count += 1
            attempts += 1
            
        if attempts == max_attempts:
            print(f"Не хватило уникальных фраз для '{label}'. Собрано: {count}")

    return texts, labels

if __name__ == "__main__":
    texts, labels = generate_dataset(500)
    
    df = pd.DataFrame({"text": texts, "label": labels})
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "topic_dataset.csv")
    
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"Сохранено: {save_path} | Всего строк: {len(df)}")