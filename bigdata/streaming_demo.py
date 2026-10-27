"""
Simulated Micro-Batch Streaming Engine for Social Media Text Analytics.
Demonstrates Spark Structured Streaming and Apache Kafka integration patterns
for high-velocity social stream processing, windowed sentiment aggregations, and toxicity alerting.
"""

import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Generator
import pandas as pd


class SocialMediaStreamSimulator:
    """
    Simulates high-velocity real-time social media post streaming
    with sliding window analytics and anomaly alerts.
    """

    def __init__(self, corpus_path: str = "data/processed/unified_social_corpus.csv"):
        self.corpus_path = corpus_path
        if os.path.exists(corpus_path):
            self.df = pd.read_csv(corpus_path)
        else:
            self.df = pd.DataFrame()

    def generate_stream_events(self, max_events: int = 50, delay_seconds: float = 0.1) -> Generator[Dict[str, Any], None, None]:
        """
        Yields simulated incoming social media posts with timestamp metadata.
        """
        if self.df.empty:
            return

        posts = self.df.to_dict(orient="records")
        current_time = datetime.now()

        for i in range(max_events):
            post = random.choice(posts).copy()
            event_time = current_time + timedelta(seconds=i * random.uniform(1.0, 5.0))
            
            event = {
                "event_id": f"evt_{i+1:05d}",
                "timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                "post_id": post.get("id"),
                "text": post.get("text"),
                "language": post.get("language"),
                "script": post.get("script"),
                "sentiment": post.get("sentiment"),
                "toxicity": post.get("toxicity"),
                "sarcasm": post.get("sarcasm"),
                "is_code_mixed": post.get("is_code_mixed"),
                "source": post.get("source"),
                "likes": random.randint(0, 450),
                "retweets": random.randint(0, 80)
            }
            yield event
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    def process_sliding_window(self, events: List[Dict[str, Any]], window_size: int = 10) -> Dict[str, Any]:
        """
        Computes windowed rolling aggregations (Spark Structured Streaming style).
        """
        if not events:
            return {}

        recent = events[-window_size:]
        total_in_window = len(recent)
        
        pos_count = sum(1 for e in recent if e["sentiment"] == "positive")
        neg_count = sum(1 for e in recent if e["sentiment"] == "negative")
        neu_count = sum(1 for e in recent if e["sentiment"] == "neutral")
        toxic_count = sum(1 for e in recent if e["toxicity"] == "toxic")
        code_mixed_count = sum(1 for e in recent if e.get("is_code_mixed", False))

        toxicity_rate = round((toxic_count / total_in_window) * 100, 1)
        alert_triggered = toxicity_rate >= 30.0  # Alert if >30% toxic in recent window

        return {
            "window_size": total_in_window,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "toxic_count": toxic_count,
            "code_mixed_count": code_mixed_count,
            "toxicity_rate_pct": toxicity_rate,
            "toxicity_alert": alert_triggered,
            "dominant_sentiment": "positive" if pos_count >= max(neg_count, neu_count) else ("negative" if neg_count >= neu_count else "neutral")
        }


if __name__ == "__main__":
    import os
    print("=== Running Simulated Spark Streaming Ingestion Demo ===")
    sim = SocialMediaStreamSimulator()
    collected = []
    for event in sim.generate_stream_events(max_events=10, delay_seconds=0.01):
        collected.append(event)
        window_stats = sim.process_sliding_window(collected, window_size=5)
        print(f"[{event['timestamp']}] Lang: {event['language']:<8} | Sent: {event['sentiment']:<8} | Toxic: {event['toxicity']:<9} | Win Toxicity: {window_stats['toxicity_rate_pct']}%")
