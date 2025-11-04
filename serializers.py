from rest_framework import serializers
from .models import TextSubmission, SentimentAnalysisResult

class SentimentAnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = SentimentAnalysisResult
        fields = ["emotion","confidence_score", "created_at"]
        read_only_fields = fields

class TextSubmissionSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    results = SentimentAnalysisResultSerializer(read_only=True)

    class Meta:
        model = TextSubmission
        fields = ["id","username", "original_text", "created_at", "results"]
        read_only_fields = ["id","created_at", "results"]
