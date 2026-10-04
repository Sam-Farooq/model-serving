"""Load profile that exercises both serving paths.

Weighted to match observed traffic: most requests are a single transaction
being scored at authorisation time, a few are nightly rescoring batches.
"""
import random

from locust import HttpUser, between, task


def instance() -> dict:
    return {
        "amount": round(random.lognormvariate(4.0, 1.3), 2),
        "hour_of_day": random.randint(0, 23),
        "merchant_risk": round(random.random(), 3),
        "account_age_days": random.randint(1, 3000),
        "txn_count_24h": random.randint(0, 40),
        "distinct_countries_7d": random.randint(1, 6),
        "is_cross_border": random.random() < 0.2,
    }


class Scoring(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(20)
    def single(self):
        with self.client.post("/predict", json={"instances": [instance()]},
                              name="/predict [inline]", catch_response=True) as r:
            if r.ok and r.json()["served_by"] != "inline":
                r.failure("single instance should not have gone to Celery")

    @task(3)
    def batch(self):
        n = random.choice([16, 64, 256])
        self.client.post("/predict", json={"instances": [instance() for _ in range(n)]},
                         name=f"/predict [celery {n}]")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
