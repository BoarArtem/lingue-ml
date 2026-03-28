import sys
import os

sys.path.append(os.path.abspath("lingue-ml"))

from inference.topic_predictor import TopicPredictor

clf = TopicPredictor()

tests = [
    "I love pizza",
    "My dog is amazing",
    "We booked a hotel",
    "I have a meeting today",
    "I play football",
    "Dinner with my family"
]

for t in tests:
    print(f"{t} -> {clf.predict(t)}")