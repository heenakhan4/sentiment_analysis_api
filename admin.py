from django.contrib import admin
from .models import TextSubmission, SentimentAnalysisResult

admin.site.register(TextSubmission)
admin.site.register(SentimentAnalysisResult)
