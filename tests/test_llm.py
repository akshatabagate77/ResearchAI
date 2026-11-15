
import pytest

import src.llm as llm


# Test 1: Read environment configuration
def test_get_config_from_environment(monkeypatch):
    monkeypatch.setenv("TEST_SETTING", "  hello  ")

    result = llm.get_config("TEST_SETTING")

    assert result == "hello"


# Test 2: Return default when configuration is missing
def test_get_config_returns_default(monkeypatch):
    monkeypatch.delenv("UNKNOWN_SETTING", raising=False)

    result = llm.get_config("UNKNOWN_SETTING", "default")

    assert result == "default"


# Test 3: Detect daily quota error
def test_is_daily_quota_error():
    assert llm.is_daily_quota_error(
        "Daily quota exceeded"
    ) is True


# Test 4: Reject non-quota error
def test_is_not_daily_quota_error():
    assert llm.is_daily_quota_error(
        "Invalid API key"
    ) is False


# Test 5: Detect temporary error
def test_is_temporary_error():
    assert llm.is_temporary_error(
        "503 Service Unavailable"
    ) is True


# Test 6: Reject non-temporary error
def test_is_not_temporary_error():
    assert llm.is_temporary_error(
        "Invalid API key"
    ) is False


# Test 7: Extract retry delay
def test_extract_retry_delay():
    result = llm.extract_retry_delay(
        '{"retryDelay": "120s"}'
    )

    assert result == 120.0


# Test 8: Handle missing retry delay
def test_extract_retry_delay_missing():
    result = llm.extract_retry_delay(
        "No retry information"
    )

    assert result is None


# Test 9: Format quota error
def test_format_quota_error():
    result = llm.format_quota_error(
        '{"retryDelay": "3600s"}'
    )

    assert "quota has been exhausted" in result
    assert "1 hours and 0 minutes" in result


# Test 10: Reject empty prompt
def test_generate_answer_rejects_empty_prompt():
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        llm.generate_answer("   ")


from types import SimpleNamespace


# Test 11: Generate a successful response
def test_generate_answer_success(monkeypatch):
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text="  Machine learning is useful.  "
            )
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )

    result = llm.generate_answer("Explain machine learning")

    assert result == "Machine learning is useful."


# Test 12: Handle an empty Gemini response
def test_generate_answer_empty_response(monkeypatch):
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text="")
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )

    with pytest.raises(llm.GeminiResponseError):
        llm.generate_answer("Explain AI")


# Test 13: Reject invalid max_retries
def test_generate_answer_invalid_retries():
    with pytest.raises(ValueError):
        llm.generate_answer("Explain AI", max_retries=-1)


# Test 14: Reject boolean max_retries
def test_generate_answer_boolean_retries():
    with pytest.raises(ValueError):
        llm.generate_answer("Explain AI", max_retries=True)


# Test 15: Reject invalid output tokens
def test_generate_answer_invalid_tokens():
    with pytest.raises(ValueError):
        llm.generate_answer("Explain AI", max_output_tokens=0)


# Test 16: Convert quota error into GeminiQuotaError
def test_generate_answer_quota_error(monkeypatch):
    def fake_generate_content(**kwargs):
        raise Exception("Daily quota exceeded")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )

    with pytest.raises(llm.GeminiQuotaError):
        llm.generate_answer("Explain AI")


# Test 17: Retry after a temporary error
def test_generate_answer_retries_temporary_error(monkeypatch):
    calls = {"count": 0}

    def fake_generate_content(**kwargs):
        calls["count"] += 1

        if calls["count"] == 1:
            raise Exception("503 Service Unavailable")

        return SimpleNamespace(text="Success")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0)

    result = llm.generate_answer(
        "Explain AI",
        max_retries=1
    )

    assert result == "Success"
    assert calls["count"] == 2


# Test 18: Handle temporary error after all retries
def test_generate_answer_retries_exhausted(monkeypatch):
    def fake_generate_content(**kwargs):
        raise Exception("503 Service Unavailable")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0)

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        llm.generate_answer(
            "Explain AI",
            max_retries=1
        )


# Test 19: Handle other API errors
def test_generate_answer_other_api_error(monkeypatch):
    def fake_generate_content(**kwargs):
        raise Exception("Invalid API key")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )

    with pytest.raises(RuntimeError, match="Gemini API error"):
        llm.generate_answer("Explain AI")


# Test 20: Raise error when API key is missing
def test_get_gemini_client_missing_api_key(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "API_KEY", None)

    with pytest.raises(ValueError, match="Gemini API key not found"):
        llm.get_gemini_client()


# Test 21: Initialize Gemini client successfully
def test_get_gemini_client_initialization(monkeypatch):
    fake_client = object()

    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "API_KEY", "fake-api-key")
    monkeypatch.setattr(
        llm.genai,
        "Client",
        lambda api_key: fake_client
    )

    result = llm.get_gemini_client()

    assert result is fake_client


# Test 22: Reuse existing Gemini client
def test_get_gemini_client_reuses_existing_client(monkeypatch):
    existing_client = object()

    monkeypatch.setattr(llm, "_client", existing_client)

    result = llm.get_gemini_client()

    assert result is existing_client

# Test 23: Read configuration from Streamlit secrets
def test_get_config_from_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("CLOUD_SETTING", raising=False)

    fake_streamlit = SimpleNamespace(
        secrets={"CLOUD_SETTING": "  cloud-value  "}
    )
    monkeypatch.setattr(llm, "st", fake_streamlit)

    result = llm.get_config("CLOUD_SETTING")

    assert result == "cloud-value"


# Test 24: Handle Streamlit secrets access failure
def test_get_config_secrets_error(monkeypatch):
    class FakeSecrets:
        def get(self, name, default=None):
            raise RuntimeError("Secrets unavailable")

    monkeypatch.delenv("MISSING_SETTING", raising=False)

    fake_streamlit = SimpleNamespace(
        secrets=FakeSecrets()
    )
    monkeypatch.setattr(llm, "st", fake_streamlit)

    result = llm.get_config("MISSING_SETTING", "fallback")

    assert result == "fallback"


# Test 25: Re-raise an existing Gemini quota error
def test_generate_answer_reraises_quota_error(monkeypatch):
    def fake_generate_content(**kwargs):
        raise llm.GeminiQuotaError("Quota exhausted")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content
        )
    )

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )

    with pytest.raises(llm.GeminiQuotaError, match="Quota exhausted"):
        llm.generate_answer("Explain AI")


# Test 26: Handle an empty retry loop
def test_generate_answer_final_fallback(monkeypatch):
    fake_client = object()

    monkeypatch.setattr(
        llm, "get_gemini_client", lambda: fake_client
    )

    monkeypatch.setattr(
        llm, "range", lambda *args: [], raising=False
    )

    with pytest.raises(
        RuntimeError,
        match="could not generate a response"
    ):
        llm.generate_answer("Explain AI")
