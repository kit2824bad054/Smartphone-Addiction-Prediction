"""
FastAPI Backend: Screen Signal — Smartphone Addiction Prediction API
Loads pre-trained model.joblib and exposes REST endpoints with CORS support.
"""

import os
import sys
from typing import Dict, List, Any
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from chat import ChatRequest, ChatResponse, generate_chat_reply
except ImportError:
    from .chat import ChatRequest, ChatResponse, generate_chat_reply


# Initialize FastAPI application
app = FastAPI(
    title="Screen Signal API",
    version="1.0.0",
    description="Production-grade API for predicting smartphone addiction risk and behavioral factors.",
)

# Step 8: Configure CORS middleware to accept requests from any origin / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Artifact Locator and Loader
def find_model_artifact() -> str:
    possible_paths = [
        "model.joblib",
        os.path.join(os.path.dirname(__file__), "model.joblib"),
        os.path.join(os.path.dirname(__file__), "..", "model.joblib"),
        os.path.join(os.path.dirname(__file__), "..", "backend", "model.joblib"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("model.joblib could not be located in standard search paths.")

model_package = None

def load_model():
    global model_package
    model_path = find_model_artifact()
    print(f"[Backend Startup] Loading model artifact from: {model_path}")
    model_package = joblib.load(model_path)
    print(f"[Backend Startup] Model successfully loaded. Classes: {model_package.get('classes')}")

# Load model immediately at module import
try:
    load_model()
except Exception as e:
    print(f"[Backend Startup Error] Could not load model: {e}")

# Recommendation Templates Library
RECOMMENDATION_TEMPLATES = {
    "screen_time_hours": {
        "title": "Screen Time Curfew",
        "tip": "Your daily screen duration is elevated. Introduce scheduled 45-minute offline work blocks and leave your device in another room during meals.",
    },
    "social_media_hours": {
        "title": "Social App Boundaries",
        "tip": "Social feeds are heavily driving addiction risk. Set a strict 45-minute daily limit on TikTok/Instagram and remove infinite-scroll apps from your home dock.",
    },
    "unlocks_per_day": {
        "title": "Tame Compulsive Checking",
        "tip": "Frequent unlocks indicate habituated micro-checking. Turn off non-essential notifications and schedule fixed checking windows (e.g., 10 AM, 2 PM, 6 PM).",
    },
    "night_usage_ratio": {
        "title": "Eliminate Pre-Bed Screen Time",
        "tip": "Late-night screen use suppresses melatonin synthesis. Institute a digital curfew 60 minutes before bed and charge your phone away from the nightstand.",
    },
    "sleep_hours": {
        "title": "Sleep Hygiene Restoration",
        "tip": "Sleep deficits erode prefrontal executive control, intensifying compulsive device cravings. Aim for 7.5+ hours of consistent sleep nightly.",
    },
    "healthy_baseline": {
        "title": "Maintain Digital Balance",
        "tip": "Your current screen and sleep patterns are well-calibrated. Keep protecting your deep work blocks and device-free morning routine.",
    },
    "preventative": {
        "title": "Mindful Friction",
        "tip": "Enable grayscale screen mode occasionally to reduce the hyper-stimulating dopamine loop of colorful app icons and banners.",
    },
}

# Request and Response Schemas
class PredictionRequest(BaseModel):
    screen_time_hours: float = Field(..., ge=0.0, le=24.0, description="Daily screen time in hours")
    unlocks_per_day: float = Field(..., ge=0.0, le=1000.0, description="Number of phone unlocks/checks per day")
    social_media_hours: float = Field(..., ge=0.0, le=24.0, description="Daily hours spent on social media")
    night_usage_ratio: float = Field(..., ge=0.0, description="Night-time usage ratio (0.0 to 1.0, or 0% to 100%)")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Nightly sleep duration in hours")

class FeatureContribution(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str

class PredictionResponse(BaseModel):
    risk_level: str
    risk_score: int
    probabilities: Dict[str, float]
    feature_contributions: List[FeatureContribution]
    recommendations: List[str]
    inputs: Dict[str, float]

@app.get("/")
def read_root():
    return {
        "app": "Screen Signal API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model_package is not None,
        "docs_url": "/docs",
    }

@app.get("/health")
def health_check():
    if model_package is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": model_package.get("classes"),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model_package is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model could not be loaded: {str(e)}",
            )

    model = model_package["model"]
    scaler = model_package["scaler"]
    features_list = model_package["features"]
    feature_labels = model_package.get("feature_labels", {})

    # Robust night_usage_ratio parsing: handle both decimal (0.05) and percentage (5.0%)
    raw_night = request.night_usage_ratio
    normalized_night = raw_night / 100.0 if raw_night > 1.0 else raw_night
    normalized_night = float(np.clip(normalized_night, 0.0, 1.0))

    # Construct input vector in strict feature order
    raw_vector = [
        request.screen_time_hours,
        request.unlocks_per_day,
        request.social_media_hours,
        normalized_night,
        request.sleep_hours,
    ]

    # Standardize input vector
    scaled_vector = scaler.transform([raw_vector])[0]

    # Softmax probabilities
    probs = model.predict_proba([scaled_vector])[0]
    classes = list(model.classes_)  # [0, 1, 2]
    class_names = model_package.get("classes", ["Low", "Moderate", "High"])

    prob_dict = {}
    for cls_idx, p in zip(classes, probs):
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        prob_dict[name] = round(float(p), 4)

    # Risk Score: 0-100 continuous score weighted toward High-risk class
    p_mod = prob_dict.get("Moderate", 0.0)
    p_high = prob_dict.get("High", 0.0)
    risk_score = int(np.clip(round(p_mod * 50.0 + p_high * 100.0), 0, 100))

    # Risk Level classification
    if risk_score >= 65 or p_high > 0.45:
        risk_level = "High"
    elif risk_score >= 35 or p_mod > 0.45:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Feature Contributions to High-Risk Class
    high_class_idx = 2  # High Risk class index
    coefs = model.coef_[high_class_idx]
    contributions = []
    positive_drivers = []

    for i, fname in enumerate(features_list):
        impact_val = float(coefs[i] * scaled_vector[i])
        direction = "risk_elevator" if impact_val > 0 else "protective"
        flabel = feature_labels.get(fname, fname)
        contributions.append(
            FeatureContribution(
                feature=fname,
                label=flabel,
                impact=round(impact_val, 2),
                direction=direction,
            )
        )
        if impact_val > 0:
            positive_drivers.append((fname, impact_val))

    # Sort positive drivers descending to pick top 2
    positive_drivers.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    if len(positive_drivers) >= 2:
        top1, top2 = positive_drivers[0][0], positive_drivers[1][0]
        recommendations.append(RECOMMENDATION_TEMPLATES[top1]["tip"])
        recommendations.append(RECOMMENDATION_TEMPLATES[top2]["tip"])
    elif len(positive_drivers) == 1:
        top1 = positive_drivers[0][0]
        recommendations.append(RECOMMENDATION_TEMPLATES[top1]["tip"])
        recommendations.append(RECOMMENDATION_TEMPLATES["preventative"]["tip"])
    else:
        recommendations.append(RECOMMENDATION_TEMPLATES["healthy_baseline"]["tip"])
        recommendations.append(RECOMMENDATION_TEMPLATES["preventative"]["tip"])

    return PredictionResponse(
        risk_level=risk_level,
        risk_score=risk_score,
        probabilities=prob_dict,
        feature_contributions=contributions,
        recommendations=recommendations,
        inputs={
            "screen_time_hours": request.screen_time_hours,
            "unlocks_per_day": request.unlocks_per_day,
            "social_media_hours": request.social_media_hours,
            "night_usage_ratio": normalized_night,
            "sleep_hours": request.sleep_hours,
        },
    )

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Real AI Digital-Wellbeing Chat endpoint powered by Anthropic Claude.
    Accepts: {message: string, risk_level: string, top_factor: string, score: number}
    Returns: {reply: string}
    """
    reply = generate_chat_reply(
        message=request.message,
        risk_level=request.risk_level,
        top_factor=request.top_factor,
        score=request.score,
    )
    return ChatResponse(reply=reply)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

