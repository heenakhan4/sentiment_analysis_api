from django.db import models
from django.contrib.auth.models import User


# Create your models here.
"""TextSubmission

user (ForeignKey to Django User)
original_text (TextField for the submitted text)
created_at (DateTimeField auto set on creation)
updated_at (DateTimeField auto updates on save)
is_archived (BooleanField, default False)
__str__ method
Add class Meta with ordering by most recent first


SentimentAnalysisResult

submission (OneToOneField to TextSubmission - one result per submission)
emotion (CharField with choices: POSITIVE, NEGATIVE, NEUTRAL)
confidence_score (FloatField for the confidence)
model_used (CharField to store which model processed it)
processing_time_ms (IntegerField for milliseconds taken)
created_at (DateTimeField)
__str__ method
Add class Meta with ordering


AnalysisCache (Optional for now - you can add this later)

"""

class TextSubmission(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_text = models.TextField(max_length=1000, blank=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_archived = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.original_text[:30]}"
    class Meta:
        ordering = ["-created_at"]

class SentimentAnalysisResult(models.Model):
    EMOTION_CHOICES = [ 
        ("POSITIVE", "Positive"),
        ("NEGATIVE","Negative"),
        ("NEUTRAL", "Neutral")
    ]

    submission = models.OneToOneField(TextSubmission, on_delete=models.CASCADE)
    emotion = models.CharField(max_length=10, choices=EMOTION_CHOICES)
    confidence_score = models.FloatField()
    model_used = models.CharField(max_length=50)
    processing_time_ms = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.submission} - {self.emotion}({self.confidence_score:.2f})"

    class Meta:
        ordering = ["-created_at"]