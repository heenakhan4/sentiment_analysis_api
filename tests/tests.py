# from django.test import TestCase, client
# import json
# import os

# data_path = os.path.join(os.path.dirname(__file__), "data", "test_Dataset.json")
# with open(data_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

import json
import os
from django.test import TestCase, Client

class SentimentAPITestCase(TestCase):
    """Tests the Sentiment Analysis API against predefined dataset."""

    @classmethod
    def setUpTestData(cls):
        # Load test dataset
        data_path = os.path.join(os.path.dirname(__file__), "data", "test_dataset.json")
        with open(data_path, "r", encoding="utf-8") as f:
            cls.dataset = json.load(f)
        cls.client = Client()

    def test_sentiment_predictions(self):
        total = len(self.dataset)
        matches = 0

        for item in self.dataset:
            text = item["text"]
            expected = item["expected_sentiment"].lower()

            # Send request to your API
            response = self.client.post(
                "/api/sentiment/",
                data=json.dumps({"text": text}),
                content_type="application/json"
            )

            self.assertEqual(response.status_code, 200, msg=f"API failed for: {text}")
            data = response.json()

            predicted = data.get("sentiment", "").lower()
            confidence = data.get("confidence", 0)

            # Optional: handle fuzzy matching (e.g., mixed/ambiguous cases)
            if expected in predicted or (
                expected == "ambiguous" and confidence < 0.6
            ):
                matches += 1
            else:
                print(f"❌ Mismatch | Text: {text}\nExpected: {expected}, Got: {predicted}\n")

        accuracy = (matches / total) * 100
        print(f"\n✅ Accuracy: {accuracy:.2f}% ({matches}/{total})")

        # Ensure at least 80% agreement
        self.assertGreaterEqual(
            accuracy, 80,
            msg=f"Accuracy below acceptable threshold: {accuracy:.2f}%"
        )
