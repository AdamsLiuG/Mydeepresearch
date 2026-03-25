"""Locust tasks used by the engineering load benchmark."""

from __future__ import annotations

import os

from locust import HttpUser, constant, task


class ResearchUser(HttpUser):
    wait_time = constant(float(os.getenv("LOCUST_WAIT_TIME", "0")))

    @task
    def run_research(self) -> None:
        payload = {"topic": os.getenv("LOCUST_RESEARCH_TOPIC", "Engineering benchmark load test")}
        search_api = os.getenv("LOCUST_SEARCH_API")
        if search_api:
            payload["search_api"] = search_api

        self.client.post(
            "/research",
            json=payload,
            name="/research",
            headers={"X-Request-ID": "locust-benchmark"},
        )
