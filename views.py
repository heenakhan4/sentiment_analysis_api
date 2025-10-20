from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.views import APIView
from django.contrib.auth.models import User
from .models import TextSubmission, SentimentAnalysisResult
from rest_framework import status
from transformers import pipeline
import time
import logging


logger = logging.getLogger(__name__)

# error function
def error(message):
    return Response({
        "success": False,
        "message": message
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# module loading function
SENTIMENT_ANALYZER = None

def load_model():
    global SENTIMENT_ANALYZER
    try:
        logger.info("Loading sentiment model")
        SENTIMENT_ANALYZER = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("Sentiment model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to laded sentiment model: {str(e)}")
        SENTIMENT_ANALYZER = None
        

# register user
@api_view(['POST'])
@permission_classes([AllowAny])

def register(request):
    try:
        username = request.data.get("username")
        password = request.data.get("password")
        
        if not username or not password:
            logger.error("Username and password are required")
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
    

# load model
load_model()

# Sentiment Analysis
class Analyze(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user = request.user
        text = request.data.get("text","").strip()

        if not text:
            logger.warning(f"User {user.username} submitted empty text")
            return error("Text is required")
        
        max_length = 5000
        if len(text) > max_length:
            logger.warning(f"User {user.username} submitted text exceeding max limit. Text length: {len(text)}")
            return error(f"Text length should be less than {max_length} characters")
        
        if not SENTIMENT_ANALYZER:
            logger.error("Sentiment model is not loaded")
            return error("Sentiment analysis model currently unavailable")
        try:
            submission = TextSubmission.objects.create(user=user, original_text=text)
            logger.info(f"Sumission created successfully with id: {submission.id}")
        except Exception as e:
            logger.error(f"Failed to create text submission: {str(e)}")
            
        start_time = time.time()
        result = SENTIMENT_ANALYZER(text)
        end_time = time.time()
        
        SentimentAnalysisResult.objects.create(
            submission = submission,
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
    
    def get(self, request):
        user = request.user
        text = TextSubmission.objects.filter(user=user)
        original_text = list(text.values('original_text'))
        all_results = SentimentAnalysisResult.objects.filter(submission__user = user)
        values =  list(all_results.values('submission','emotion', 'confidence_score', 'created_at'))

        results = {}
        for i in range(len(values)):
            results[values[i]['submission']] = {
                "text": original_text[i]['original_text'],
                "emotion": values[i]['emotion'],
                "confidence_score": values[i]['confidence_score'],
                "created_at": values[i]['created_at']
            }

        return Response({
            "success": True,
            "message": "Fecthed analysis data",
            "data": {
                "username": user.username,
                "results": results
            }
        })