"""
Chat Module: Real AI Digital-Wellbeing Assistant powered by Anthropic Claude.
Loads ANTHROPIC_API_KEY from .env using python-dotenv and handles requests gracefully.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import anthropic

# Look for .env in current working directory, backend folder, and project root
_env_locations = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]

for p in _env_locations:
    if p.is_file():
        load_dotenv(dotenv_path=p, override=True)
        break
else:
    load_dotenv()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's query or message")
    risk_level: str = Field(..., description="Current assessed risk level: Low, Moderate, or High")
    top_factor: str = Field(..., description="Top contributing habit factor, e.g. Bedtime Screen, Phone Unlocks")
    score: float = Field(..., ge=0.0, le=100.0, description="Risk assessment score (0 to 100)")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Previous messages in the conversation")


class ChatResponse(BaseModel):
    reply: str


def generate_smart_fallback(message: str, risk_level: str, top_factor: str, score: float) -> str:
    """Provides high-quality, practical digital wellbeing suggestions if the API key is not yet configured."""
    msg = message.lower()

    if any(w in msg for w in ["night", "bed", "sleep", "scroll", "sunset", "melatonin", "evening"]):
        return (
            "**Nighttime Screen Interventions**:\n\n"
            "1. **60-Minute Digital Sunset**: Cease all screen use 1 hour prior to sleep so melatonin can rise naturally.\n"
            "2. **Charging Displacement**: Plug your charger in another room or at least 10 feet away from your bed.\n"
            "3. **Analog Bedside Table**: Keep a physical book, journal, or standalone alarm clock near your bed instead of your phone."
        )
    elif any(w in msg for w in ["unlock", "check", "pick", "compulsive", "reflex", "habit", "twitch"]):
        return (
            "**Compulsive Pickup Interventions**:\n\n"
            "1. **Notification Cleanse**: Turn off all banners and badges except direct human phone calls.\n"
            "2. **3 Scheduled Windows**: Allocate dedicated 15-minute checking slots (e.g. 10 AM, 2 PM, 6 PM) rather than continuous micro-glances.\n"
            "3. **Physical Friction**: Put an elastic band around your phone or use a complex alphanumeric passcode to disrupt mindless muscle memory."
        )
    elif any(w in msg for w in ["social", "instagram", "tiktok", "reels", "shorts", "feed", "doomscroll"]):
        return (
            "**Social Media Boundary Protocol**:\n\n"
            "1. **Grayscale Display**: Switch your phone display to black & white (Settings > Accessibility > Color Filters). Without color, feeds lose up to 40% of their draw.\n"
            "2. **30-Minute App Timer**: Enforce a strict daily cap in Digital Wellbeing / Screen Time.\n"
            "3. **Browser Only**: Delete native apps and access feeds only via mobile browser to add intentional friction."
        )
    elif any(w in msg for w in ["screen", "time", "hour", "reduce", "cut", "focus", "work"]):
        return (
            f"**Action Plan for {risk_level} Risk ({score}/100)**:\n\n"
            "1. **Out-of-Sight Work Sprints**: Place your phone in a desk drawer during 50-minute deep work intervals.\n"
            "2. **Device-Free Morning**: Protect the first 30 minutes after waking up from all screen exposure.\n"
            "3. **Analog Substitutions**: Fill micro-breaks with physical stretching, walking, or hydration instead of screen scrolling."
        )
    else:
        return (
            f"**Digital Wellbeing Coach Recommendation**:\n\n"
            f"Based on your assessment ({risk_level} Risk, {score}/100 score, top driver: {top_factor}), "
            "small environmental changes work better than willpower:\n\n"
            "• **Home Screen Cleanse**: Remove distracting social and video apps from your main dock.\n"
            "• **Nightly Displacement**: Charge your phone across the room or outside the bedroom.\n"
            "• **Batch Notifications**: Schedule fixed daily check-in blocks to protect deep work."
        )


def call_gemini_api(api_key: str, message: str, system_prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Calls Google Gemini API using fast and reliable candidate models with conversation history."""
    candidate_models = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
    ]
    headers = {"Content-Type": "application/json"}
    
    contents = []
    if history:
        for turn in history[-8:]:
            raw_role = str(turn.get("role", "")).lower()
            role = "model" if raw_role in ["assistant", "model", "bot"] else "user"
            text_val = str(turn.get("content") or turn.get("text") or "").strip()
            if text_val:
                contents.append({
                    "role": role,
                    "parts": [{"text": text_val}]
                })

    contents.append({
        "role": "user",
        "parts": [{"text": message.strip()}]
    })

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 1000,
            "temperature": 0.85
        }
    }

    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    reply = "".join(p.get("text", "") for p in parts if not p.get("thought", False)).strip()
                    if reply:
                        return reply
        except Exception:
            continue

    return ""


def generate_chat_reply(message: str, risk_level: str, top_factor: str, score: float, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Calls Google Gemini API (or Anthropic Claude API as fallback) with a supportive digital-wellbeing system prompt.
    Includes past conversation turns and explicit instructions to prevent repetitive answers.
    """
    # Reload .env if modified during runtime
    for p in _env_locations:
        if p.is_file():
            load_dotenv(dotenv_path=p, override=True)

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # Detect if user placed Gemini key in ANTHROPIC_API_KEY
    if not gemini_key and (anthropic_key.startswith("AQ.") or anthropic_key.startswith("AIza")):
        gemini_key = anthropic_key

    # Build system prompt with anti-repetition directive
    system_prompt = (
        f"You are an encouraging, practical digital-wellbeing AI coach. "
        f"The user's smartphone risk assessment: "
        f"risk_level={risk_level}, score={score}/100, top driver={top_factor}. "
        f"Answer the user's latest query directly with actionable, compassionate guidance. "
        f"CRITICAL RULE: Do NOT repeat the exact same suggestions, bullet points, or examples you have already given in this conversation. "
        f"Introduce fresh strategies, psychological friction ideas, habit-stacking substitutions, or reflection exercises every time. "
        f"Keep replies clear, engaging, and under 110 words."
    )

    # 1. Try Google Gemini API if Gemini key is available
    if gemini_key and not gemini_key.startswith("your_") and gemini_key != "placeholder":
        reply = call_gemini_api(gemini_key, message, system_prompt, history=history)
        if reply:
            return reply

    # 2. Try Anthropic Claude API if Anthropic key is available (starts with sk-)
    api_key = anthropic_key
    if not api_key or api_key.startswith("your_") or api_key == "placeholder" or api_key.startswith("AQ.") or api_key.startswith("AIza"):
        advice = generate_smart_fallback(message, risk_level, top_factor, score)
        return advice

    # Candidate models to try (latest sonnet / configurable)
    configured_model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest").strip()
    candidate_models = [
        configured_model,
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ]

    unique_models = []
    for m in candidate_models:
        if m and m not in unique_models:
            unique_models.append(m)

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        return f"⚠️ Anthropic client initialization failed: {str(e)}"

    last_error = None
    for model_name in unique_models:
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=300,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": message.strip()}
                ],
            )
            # Concatenate text content blocks
            reply_text = "".join(
                getattr(block, "text", "") for block in response.content if hasattr(block, "text")
            ).strip()

            if reply_text:
                return reply_text

        except anthropic.NotFoundError as e:
            # Model not available in account tier, try next fallback model
            last_error = e
            continue
        except anthropic.AuthenticationError:
            fallback = generate_smart_fallback(message, risk_level, top_factor, score)
            return (
                f"{fallback}\n\n"
                "*(⚠️ Note: The provided `ANTHROPIC_API_KEY` in `.env` is invalid. Showing offline guidance).* "
            )
        except anthropic.RateLimitError:
            fallback = generate_smart_fallback(message, risk_level, top_factor, score)
            return (
                f"{fallback}\n\n"
                "*(⚠️ Note: Anthropic rate limit reached. Showing offline guidance).* "
            )
        except anthropic.APIConnectionError:
            fallback = generate_smart_fallback(message, risk_level, top_factor, score)
            return (
                f"{fallback}\n\n"
                "*(⚠️ Note: Could not reach Anthropic servers. Showing offline guidance).* "
            )
        except anthropic.BadRequestError as e:
            return f"⚠️ Anthropic Request Error: {e.message}"
        except anthropic.APIStatusError as e:
            return f"⚠️ Anthropic API Error ({e.status_code}): {e.message}"
        except Exception as e:
            return f"⚠️ AI Assistant Error: {str(e)}"

    if last_error:
        fallback = generate_smart_fallback(message, risk_level, top_factor, score)
        return f"{fallback}\n\n*(Model note: {str(last_error)})*"

    return generate_smart_fallback(message, risk_level, top_factor, score)
