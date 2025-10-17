from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from django.contrib.auth.models import User
from .models import TextSubmission, SentimentAnalysisResult
from rest_framework import status
from transformers import pipeline
import time

# error function
def error(message):
    return Response({
        "success": False,
        "message": message
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# register user
@api_view(['POST'])
def register(request):
    try:
        
        username = request.data.get("username")
        password = request.data("password")
        print(username, password)
        if not username or not password:
            return Response({
                "success": False,
                "message": "Username and password are required"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if User.objects.filter(username=username).exists():
            return Response({
                "success": False,
                "message": "User already exists"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        User.objects.create_user(username=username, password=password)

        return Response({
            "success": True,
            "message": "User registered successfully"
        },status=status.HTTP_201_CREATED)

    except Exception as e:
        return error(str(e))


# Sentiement Analysis
class Analyze(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user = request.user
        text = request.data.get("text")

        if not text:
            return error("Text is required")
        
        TextSubmission.objects.create(user=user, original_text=text)
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        start_time = time.time()
        result = sentiment_analyzer(text)
        end_time = time.time()
        
        SentimentAnalysisResult.objects.create(
            submission = TextSubmission.objects.get(user=user, original_text=text),
            emotion = result[0]["label"],
            confidence_score = result[0]["score"],
            model_used = "distilbert-base-uncased-finetuned-sst-2-english",
            processing_time_ms = int((end_time - start_time)*1000)
        )
        
        return Response({
            "success": True,
            "message": "Text analyzed successfully",
            "data": {
                "username": user.username,
                "text": text,
                "result": result
            }
        },status=status.HTTP_200_OK)