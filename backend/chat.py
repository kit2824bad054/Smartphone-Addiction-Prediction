"""
Chat Module: Real AI Digital-Wellbeing Assistant powered by Anthropic Claude.
Loads ANTHROPIC_API_KEY from .env using python-dotenv and handles requests gracefully.
"""

import os
from pathlib import Path
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


class ChatResponse(BaseModel):
    reply: str


def generate_chat_reply(message: str, risk_level: str, top_factor: str, score: float) -> str:
    """
    Calls the Anthropic Claude API with a supportive digital-wellbeing system prompt.
    Gracefully handles missing API keys, rate limits, model availability, and network issues.
    """
    # Reload .env if it was created/modified during runtime
    for p in _env_locations:
        if p.is_file():
            load_dotenv(dotenv_path=p, override=False)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # Graceful handling of missing or placeholder API key
    if not api_key or api_key.startswith("your_") or api_key == "placeholder":
        return (
            "⚠️ Anthropic API key is not configured. "
            "Please add ANTHROPIC_API_KEY to your .env file in the backend directory to enable live AI responses."
        )

    # Build system prompt according to requirements
    system_prompt = (
        f"You are a supportive digital-wellbeing assistant. "
        f"The user's smartphone risk assessment shows: "
        f"risk_level={risk_level}, score={score}/100, top contributing factor={top_factor}. "
        f"Give brief, practical, non-judgmental suggestions to reduce screen time related to their situation. "
        f"Keep replies under 100 words."
    )

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
            return (
                "⚠️ Authentication Error: The provided Anthropic API key is invalid. "
                "Please verify your ANTHROPIC_API_KEY in the .env file."
            )
        except anthropic.RateLimitError:
            return (
                "⚠️ Rate Limit: Anthropic API rate limit reached. "
                "Please wait a few moments before trying again."
            )
        except anthropic.APIConnectionError:
            return (
                "⚠️ Connection Error: Unable to reach Anthropic API servers. "
                "Please check your internet connection."
            )
        except anthropic.BadRequestError as e:
            return f"⚠️ Anthropic Request Error: {e.message}"
        except anthropic.APIStatusError as e:
            return f"⚠️ Anthropic API Error ({e.status_code}): {e.message}"
        except Exception as e:
            return f"⚠️ AI Assistant Error: {str(e)}"

    if last_error:
        return f"⚠️ Anthropic Model Error: {str(last_error)}"

    return "⚠️ Could not generate an AI response. Please try again."
