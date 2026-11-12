

import os
import re
import time
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ---------------------------------------------------------
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ---------------------------------------------------------
# STEP 2: CUSTOM EXCEPTIONS
# ---------------------------------------------------------

class GeminiQuotaError(Exception):
    """
    Raised when Gemini API quota is exhausted.
    """
    pass


class GeminiResponseError(Exception):
    """
    Raised when Gemini returns an invalid or empty response.
    """
    pass


# ---------------------------------------------------------
# STEP 3: INITIALIZE GEMINI CLIENT
# ---------------------------------------------------------

_client = None


def get_gemini_client():
    """
    Initialize Gemini client only when required.

    This prevents application startup from failing
    just because the API key is missing.
    """

    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found.\n"
            "Please add GEMINI_API_KEY to your .env file."
        )

    _client = genai.Client(api_key=api_key)

    return _client


# ---------------------------------------------------------
# STEP 4: DETECT DAILY QUOTA ERRORS
# ---------------------------------------------------------

def is_daily_quota_error(error_message):
    """
    Identify daily or project-level quota exhaustion.

    These errors should not be retried immediately.
    """

    message = error_message.lower()

    daily_indicators = [
        "generate_content_free_tier_requests",
        "requests per day",
        "request per day",
        "per_day",
        "per day limit exceeded",
        "daily quota",
        "daily limit",
        "quota exceeded for quota metric",
        "free_tier_requests_per_day"
    ]

    return any(
        indicator in message
        for indicator in daily_indicators
    )


# ---------------------------------------------------------
# STEP 5: DETECT TEMPORARY ERRORS
# ---------------------------------------------------------

def is_temporary_error(error_message):
    """
    Identify errors that may recover after waiting.
    """

    message = error_message.lower()

    temporary_indicators = [
        "503",
        "service unavailable",
        "500",
        "internal server error",
        "429",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "requests per minute",
        "tokens per minute",
        "temporarily unavailable",
        "deadline exceeded",
        "timeout"
    ]

    return any(
        indicator in message
        for indicator in temporary_indicators
    )


# ---------------------------------------------------------
# STEP 6: EXTRACT GEMINI RETRY DELAY
# ---------------------------------------------------------

def extract_retry_delay(error_message):
    """
    Extract retry delay from Gemini error messages.

    Example:
        retryDelay: '7984s'
    """

    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
        error_message,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None


# ---------------------------------------------------------
# STEP 7: FORMAT QUOTA ERROR MESSAGE
# ---------------------------------------------------------

def format_quota_error(error_message):
    """
    Generate a useful message when quota is exhausted.
    """

    retry_delay = extract_retry_delay(error_message)

    message = (
        "Your Gemini API quota has been exhausted.\n\n"
        "This may be a daily or project-level quota limit. "
        "Immediate retries will not solve this problem."
    )

    if retry_delay is not None:

        hours = int(retry_delay // 3600)

        minutes = int(
            (retry_delay % 3600) // 60
        )

        message += (
            f"\n\nGemini suggests waiting approximately "
            f"{hours} hours and {minutes} minutes."
        )

    message += (
        "\n\nCheck your Gemini API usage, model limits "
        "and billing settings before trying again."
    )

    return message


# ---------------------------------------------------------
# STEP 8: GENERATE AI RESPONSE
# ---------------------------------------------------------

def generate_answer(
    prompt,
    max_retries=2,
    temperature=0.3,
    max_output_tokens=2000
):
    """
    Generate a response using Google Gemini.

    Features:
        1. Lazy client initialization.
        2. Daily quota detection.
        3. Temporary error retries.
        4. Exponential backoff.
        5. Empty response validation.

    Compatible with:
        generate_answer(prompt)
    """

    if not isinstance(prompt, str) or not prompt.strip():

        raise ValueError(
            "Prompt cannot be empty."
        )

    if not isinstance(max_retries, int) or max_retries < 0:

        raise ValueError(
            "max_retries must be a non-negative integer."
        )

    if max_output_tokens <= 0:

        raise ValueError(
            "max_output_tokens must be greater than zero."
        )

    # Initialize client
    client = get_gemini_client()

    # Retry loop
    for attempt in range(max_retries + 1):

        try:

            print(
                f"ResearchAI: Generating response using {MODEL_NAME}..."
            )

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt,

                config=types.GenerateContentConfig(

                    temperature=temperature,

                    max_output_tokens=max_output_tokens

                )
            )

            # Validate response
            answer = response.text

            if not answer or not answer.strip():

                raise GeminiResponseError(
                    "Gemini returned an empty response. "
                    "Try modifying your prompt."
                )

            print("ResearchAI: Response generated successfully.")

            return answer.strip()

        except GeminiQuotaError:
            raise

        except GeminiResponseError:
            raise

        except Exception as error:

            error_message = str(error)

            # CASE 1: Daily quota exhausted
            if is_daily_quota_error(error_message):

                raise GeminiQuotaError(
                    format_quota_error(error_message)
                ) from error

            # CASE 2: Temporary API error
            if is_temporary_error(error_message):

                if attempt >= max_retries:

                    raise RuntimeError(
                        "Gemini is temporarily unavailable "
                        f"after {max_retries} retries.\n\n"
                        f"Original error: {error_message}"
                    ) from error

                # Exponential backoff
                wait_time = min(
                    (2 ** (attempt + 1))
                    + random.uniform(0, 1),
                    30
                )

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time:.1f} seconds..."
                )

                time.sleep(wait_time)

                continue

            # CASE 3: Other API errors
            raise RuntimeError(
                f"Gemini API error: {error_message}"
            ) from error

    raise RuntimeError(
        "ResearchAI could not generate a response."
    )

