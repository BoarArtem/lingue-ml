from locust import HttpUser, task, between
import random

WORDS = ["Bread", "Water", "Apple", "Table", "Computer"]
LANGUAGES = ["english", "german", "ukrainian"]
LEVELS = ["A1", "A2", "B1", "B2"]

class APIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def sentence(self):
        payload = {
            "word": random.choice(WORDS),
            "language": random.choice(LANGUAGES),
            "level": random.choice(LEVELS)
        }

        self.client.post(
            "/sentence",
            json=payload
        )