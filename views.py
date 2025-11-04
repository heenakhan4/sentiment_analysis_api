from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.views import APIView
from django.contrib.auth.models import User
from .models import TextSubmission, SentimentAnalysisResult
from .serializers import *
from rest_framework import status
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import time
import logging
from django.db import connection
from django.utils import timezone
import torch


logger = logging.getLogger(__name__)

# error function
def error(message):
    return Response({
        "success": False,
        "message": message
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# module loading function
TOKENIZER, MODEL = None, None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_id):
    global TOKENIZER, MODEL
    try:
        logger.info("Loading sentiment model")
        
        TOKENIZER = DistilBertTokenizer.from_pretrained(model_id)
        MODEL = DistilBertForSequenceClassification.from_pretrained(model_id)
        
        MODEL.to(DEVICE)

        logger.info(f"Sentiment model loaded successfully on device: {DEVICE}")
    
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {str(e)}")
        TOKENIZER, MODEL = None, None
        

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
load_model(model_id="joeddav/distilbert-base-uncased-go-emotions-student")

# Sentiment Analysis
class Analyze(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        try:
            user = request.user
            text = request.data.get("text","").strip()
            type = request.data.get("type","")
            logger.info(f"User {user.username} submitted text for analysis")

            if not text:
                logger.warning(f"User {user.username} submitted empty text")
                return error("Text is required")
            
            max_length = 5000
            if len(text) > max_length:
                logger.warning(f"User {user.username} submitted text exceeding max limit. Text length: {len(text)}")
                return error(f"Text length should be less than {max_length} characters")
            
            if not TOKENIZER or not MODEL:
                logger.error("Sentiment model is not loaded")
                return error("Sentiment analysis model currently unavailable")
            try:
                submission = TextSubmission.objects.create(user=user, original_text=text)
                logger.info(f"Sumission created successfully with id: {submission.id}")
            except Exception as e:
                logger.error(f"Failed to create text submission: {str(e)}")
                
            try:
                start_time = time.time()
                # result = SENTIMENT_ANALYZER(text)
                # Tokenize input
                inputs = TOKENIZER(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move inputs to same device as model
                inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
                
                # Step 2: Run inference (no gradient calculation needed)
                with torch.no_grad():
                    outputs = MODEL(**inputs)
                
                # ------ MULTI-CLASS CLASSIFICATION ------
                # Step 3: Get logits and convert to probabilities
                logits = outputs.logits
                if type=="multiclass":
                    probabilities = torch.softmax(logits, dim=-1)

                    # Step 4: Get predicted class and confidence
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence_score = probabilities[0][predicted_class].item()
                
                    # Step 5: Map class to emotion label
                    emotion = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
                else:
                    probabilities = torch.sigmoid(logits)
                    print(probabilities)
                    labels = [MODEL.config.id2label[i] for i in range(MODEL.config.num_labels)]
                    print(labels)
                    threshold = 0.5
                    predicted = (probabilities[0] > threshold).nonzero(as_tuple=True)[0]
                    emotion = [labels[i] for i in predicted]
                    confidence_score = [probabilities[0][i].item() for i in predicted]
                    for label, score in zip(emotion, confidence_score):
                        print(f"{label}: {score:.4f}")

                
                end_time = time.time()
                logger.info(f"Sentiment analysis completed for submission with id: {submission.id}")
            except Exception as e:
                logger.error(f"Unexpected error occured in sentiment analysis: {str(e)}. Deleting submission with id: {submission.id}")
                submission.delete()
                return error(message = str(e))
            
            SentimentAnalysisResult.objects.create(
                submission = submission,
                emotion = emotion,
                confidence_score = confidence_score[0],
                model_used = "distilbert-base-uncased-finetuned-sst-2-english",
                processing_time_ms = int((end_time - start_time)*1000)
            )

            return Response({
                "success": True,
                "message": "Text analyzed successfully",
                "data": {
                    "username": user.username,
                    "result": {
                        "label": emotion, 
                        "score": confidence_score
                    }
                }
            },status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Unexpected error occured in sentiment analysis: {str(e)}")
            return error(message = str(e))
    
    def get(self, request):
        user = request.user
        logger.info(f"User {user.username} requested analysis data")
        text = TextSubmission.objects.filter(user=user)
        original_text = list(text.values('original_text'))
        all_results = SentimentAnalysisResult.objects.filter(submission__user = user)
        if not all_results:
            logger.info(f"Sentiment analysis QuerySet is empty for {user.username}")
            return Response({
                "success":True,
                "message": f"No analysis data found for {user.username}"
                }, status=status.HTTP_200_OK)
        
        values =  list(all_results.values('submission','emotion', 'confidence_score', 'created_at'))

        results = {}
        for i in range(len(values)):
            results[values[i]['submission']] = {
                "text": original_text[i]['original_text'],
                "emotion": values[i]['emotion'],
                "confidence_score": values[i]['confidence_score'],
                "created_at": values[i]['created_at']
            }
        logger.info(f"Successfully fetched analysis history for {user.username}")
        return Response({
            "success": True,
            "message": "Fecthed analysis data",
            "data": {
                "username": user.username,
                "results": results
            }
        }, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([AllowAny])
def health(request):
    logger.info("Checking server health")
    start_time = time.time()
    
    health = {
        "status": "ok",
        "checks":{}
    }

    # app check
    health["checks"]["app"] = "running"

    # db connection status
    try:
        connection.ensure_connection()
        health["checks"]["db"] = "connected"
        logger.info("Database connection -- OK")
    except Exception as e:
        health["status"] = "degraded"
        health["checks"]["db"] = f"error: {str(e)}"
        logger.warning(f"Error in database connection. Status: {health['checks']['db']}")

    # HuggingFace model status
    try:
        if TOKENIZER is not None and MODEL is not None:
            health["checks"]["model"] = "loaded"
            logger.info("HuggingFace Sentiment model -- OK")
        else:
            health["status"] = "degraded"
            health["checks"]["model"] = "not loaded"
            logger.warning("HuggingFace Sentiment model -- NOT OK")
    except Exception as e:
        health["status"] = "degraded"
        health["checks"]["model"] = f"error: {str(e)}"
        logger.warning(f"Error in HuggingFace Sentiment model. Status: {health['checks']['model']}")

    health["response_time_ms"] = int((time.time() - start_time)*1000)
    health["timestamp"] = timezone.now().isoformat()
    logger.info("Health check completed")

    return Response({
        "success": True,
        "health": health
    }, status=status.HTTP_200_OK if health["status"] == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE)
    
    

    