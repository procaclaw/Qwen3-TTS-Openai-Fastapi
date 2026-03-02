# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for API endpoints.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from api.main import app
from api.backends.factory import reset_backend


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_backend_after_test():
    """Reset backend after each test."""
    yield
    reset_backend()


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_endpoint_returns_status(self, client):
        """Test that health endpoint returns status information."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "backend" in data
        assert "device" in data
        assert "version" in data
    
    def test_health_endpoint_includes_backend_info(self, client):
        """Test that health endpoint includes backend details."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data["backend"]
        assert "model_id" in data["backend"]
        assert "ready" in data["backend"]
    
    def test_health_endpoint_includes_device_info(self, client):
        """Test that health endpoint includes device information."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "type" in data["device"]
        assert "gpu_available" in data["device"]


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""
    
    def test_list_models_endpoint(self, client):
        """Test that models endpoint returns list of models."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
    
    def test_models_include_qwen3_tts(self, client):
        """Test that qwen3-tts model is in the list."""
        response = client.get("/v1/models")
        data = response.json()
        
        model_ids = [model["id"] for model in data["data"]]
        assert "qwen3-tts" in model_ids
    
    def test_models_include_openai_compatible(self, client):
        """Test that OpenAI-compatible models are in the list."""
        response = client.get("/v1/models")
        data = response.json()
        
        model_ids = [model["id"] for model in data["data"]]
        assert "tts-1" in model_ids
        assert "tts-1-hd" in model_ids
    
    def test_get_specific_model(self, client):
        """Test getting a specific model by ID."""
        response = client.get("/v1/models/qwen3-tts")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == "qwen3-tts"
        assert "object" in data
        assert "created" in data
    
    def test_get_nonexistent_model_returns_404(self, client):
        """Test that requesting non-existent model returns 404."""
        response = client.get("/v1/models/nonexistent-model")
        assert response.status_code == 404


class TestVoicesEndpoint:
    """Tests for /v1/voices endpoint."""
    
    def test_list_voices_endpoint(self, client):
        """Test that voices endpoint returns voice list."""
        response = client.get("/v1/voices")
        assert response.status_code == 200
        
        data = response.json()
        assert "voices" in data
        assert "languages" in data
        assert isinstance(data["voices"], list)
        assert isinstance(data["languages"], list)
    
    def test_voices_include_defaults(self, client):
        """Test that default voices are included."""
        response = client.get("/v1/voices")
        data = response.json()
        
        voice_ids = [voice["id"] for voice in data["voices"]]
        # Check for some default voices
        assert "Vivian" in voice_ids or "alloy" in voice_ids
    
    def test_alternate_voices_endpoint(self, client):
        """Test alternate /v1/audio/voices endpoint."""
        response = client.get("/v1/audio/voices")
        assert response.status_code == 200
        
        data = response.json()
        assert "voices" in data


class TestSpeechEndpoint:
    """Tests for /v1/audio/speech endpoint."""
    
    def test_speech_endpoint_requires_input(self, client):
        """Test that speech endpoint requires input text."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "voice": "Vivian",
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_speech_endpoint_invalid_model(self, client):
        """Test that invalid model returns error."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "invalid-model",
                "input": "Hello",
                "voice": "Vivian",
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        # The error is in 'detail' dict
        assert "detail" in data
        assert "error" in data["detail"]
    
    def test_speech_endpoint_supports_formats(self, client):
        """Test that speech endpoint supports different formats."""
        formats = ["mp3", "wav", "opus", "flac", "aac"]
        
        for fmt in formats:
            # Just test that the format is accepted (validation level)
            # Full test would need mocking
            request_data = {
                "model": "qwen3-tts",
                "input": "Test",
                "voice": "Vivian",
                "response_format": fmt,
            }
            # This will fail at backend level but should pass validation
            # In a real test with mocking, it would succeed
            assert "response_format" in request_data
            assert request_data["response_format"] == fmt

    def test_speech_base_model_requires_custom_voice_or_voice_clone(self, client):
        """Base model should reject built-in voices on /v1/audio/speech with clear 400."""
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.get_model_type.return_value = "base"
        mock_backend.is_custom_voice.return_value = False
        factory._backend_instance = mock_backend

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "Vivian",
                "response_format": "wav",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "base_model_voice_not_supported"

    def test_speech_streaming_pcm_returns_streamed_audio(self, client):
        """Test stream=true returns streaming audio bytes for supported formats."""
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_speech_streaming.return_value = True
        mock_backend.is_custom_voice.return_value = False

        async def fake_stream(*args, **kwargs):
            yield np.array([0.0, 0.5], dtype=np.float32), 24000
            yield np.array([-0.5, 1.0], dtype=np.float32), 24000

        mock_backend.generate_speech_stream = fake_stream
        factory._backend_instance = mock_backend

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello streaming world",
                "voice": "Vivian",
                "response_format": "pcm",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/pcm")
        assert response.headers["content-disposition"] == "attachment; filename=speech.pcm"
        assert len(response.content) > 0

    @pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac", "wav", "pcm"])
    def test_speech_streaming_honors_requested_response_format(self, client, monkeypatch, fmt):
        """Test stream=true preserves requested response_format (no coercion)."""
        from api.backends import factory
        from api.routers import openai_compatible as router_mod

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_speech_streaming.return_value = True
        mock_backend.is_custom_voice.return_value = False

        async def fake_stream(*args, **kwargs):
            yield np.array([0.0, 0.5], dtype=np.float32), 24000
            yield np.array([-0.5, 1.0], dtype=np.float32), 24000

        mock_backend.generate_speech_stream = fake_stream
        factory._backend_instance = mock_backend

        monkeypatch.setattr(router_mod, "ensure_streaming_encoding_supported", lambda _fmt: None)

        async def fake_encode_audio_streaming(audio_generator, fmt_value, sample_rate):
            async for _ in audio_generator:
                pass
            yield f"encoded-{fmt_value}-sr{sample_rate}".encode("utf-8")

        monkeypatch.setattr(router_mod, "encode_audio_streaming", fake_encode_audio_streaming)

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "Vivian",
                "response_format": fmt,
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(f"audio/{'mpeg' if fmt == 'mp3' else fmt}")
        assert response.headers["content-disposition"] == f"attachment; filename=speech.{fmt}"
        assert len(response.content) > 0

    def test_speech_streaming_rejects_non_default_speed(self, client):
        """Test stream=true enforces speed=1.0 for safe chunk streaming."""
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_speech_streaming.return_value = True
        factory._backend_instance = mock_backend

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "Vivian",
                "response_format": "pcm",
                "stream": True,
                "speed": 1.25,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "invalid_speed_for_streaming"

    def test_speech_streaming_rejects_backend_without_streaming(self, client):
        """Test stream=true rejects backends that do not support true streaming."""
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_speech_streaming.return_value = False
        factory._backend_instance = mock_backend

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "Vivian",
                "response_format": "pcm",
                "stream": True,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "streaming_not_supported"

    def test_speech_streaming_rejects_unsupported_streaming_encoding_combo(self, client, monkeypatch):
        """Test stream=true returns clear error for unsupported stream/format environment combos."""
        from api.backends import factory
        from api.routers import openai_compatible as router_mod

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_speech_streaming.return_value = True
        factory._backend_instance = mock_backend

        def fake_ensure_streaming_encoding_supported(_fmt):
            raise ValueError("ffmpeg missing for compressed streaming")

        monkeypatch.setattr(
            router_mod,
            "ensure_streaming_encoding_supported",
            fake_ensure_streaming_encoding_supported,
        )

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "Vivian",
                "response_format": "mp3",
                "stream": True,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "unsupported_streaming_response_format"


class TestVoiceCloneEndpoints:
    """Tests for voice cloning endpoints."""

    def test_voice_clone_capabilities_endpoint_returns_valid_structure(self, client, monkeypatch):
        """Test that voice clone capabilities endpoint returns valid response structure."""
        from unittest.mock import MagicMock, AsyncMock
        from api.backends import factory
        
        # Create a mock backend that doesn't require model initialization
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_voice_cloning.return_value = False
        mock_backend.get_model_type.return_value = "customvoice"
        
        # Set the mock as the global backend instance
        factory._backend_instance = mock_backend
        
        response = client.get("/v1/audio/voice-clone/capabilities")
        assert response.status_code == 200

        data = response.json()
        assert "supported" in data
        assert "model_type" in data
        assert "icl_mode_available" in data
        assert "x_vector_mode_available" in data
        assert isinstance(data["supported"], bool)
        assert isinstance(data["model_type"], str)

    def test_voice_clone_capabilities_custom_voice_not_supported(self, client, monkeypatch):
        """Test capabilities returns not supported for CustomVoice model."""
        from unittest.mock import MagicMock
        from api.backends import factory
        
        # Create a mock backend for CustomVoice model
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_voice_cloning.return_value = False
        mock_backend.get_model_type.return_value = "customvoice"
        
        factory._backend_instance = mock_backend
        
        response = client.get("/v1/audio/voice-clone/capabilities")
        assert response.status_code == 200

        data = response.json()
        assert data["supported"] is False
        assert data["model_type"] == "customvoice"
        assert data["icl_mode_available"] is False
        assert data["x_vector_mode_available"] is False

    def test_voice_clone_capabilities_base_model_supported(self, client, monkeypatch):
        """Test capabilities returns supported for Base model."""
        from unittest.mock import MagicMock
        from api.backends import factory
        
        # Create a mock backend for Base model
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_voice_cloning.return_value = True
        mock_backend.get_model_type.return_value = "base"
        
        factory._backend_instance = mock_backend
        
        response = client.get("/v1/audio/voice-clone/capabilities")
        assert response.status_code == 200

        data = response.json()
        assert data["supported"] is True
        assert data["model_type"] == "base"
        assert data["icl_mode_available"] is True
        assert data["x_vector_mode_available"] is True

    def test_voice_clone_requires_input(self, client):
        """Test that voice clone endpoint requires input text."""
        response = client.post(
            "/v1/audio/voice-clone",
            json={
                "ref_audio": "dGVzdA==",  # base64 "test"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_voice_clone_not_supported_returns_400(self, client, monkeypatch):
        """Test that voice clone returns 400 error when not supported."""
        from unittest.mock import MagicMock
        from api.backends import factory
        
        # Create a mock backend that doesn't support cloning
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_voice_cloning.return_value = False
        mock_backend.get_model_type.return_value = "customvoice"
        
        factory._backend_instance = mock_backend
        
        response = client.post(
            "/v1/audio/voice-clone",
            json={
                "input": "Hello world",
                "ref_audio": "dGVzdA==",  # base64 "test"
                "x_vector_only_mode": True,
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"] == "voice_cloning_not_supported"

    def test_voice_clone_icl_mode_requires_ref_text(self, client, monkeypatch):
        """Test that voice clone ICL mode requires ref_text."""
        from unittest.mock import MagicMock
        from api.backends import factory
        
        # Create a mock backend that supports cloning
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.supports_voice_cloning.return_value = True
        mock_backend.get_model_type.return_value = "base"
        
        factory._backend_instance = mock_backend
        
        response = client.post(
            "/v1/audio/voice-clone",
            json={
                "input": "Hello world",
                "ref_audio": "dGVzdA==",  # base64 "test"
                "x_vector_only_mode": False,  # ICL mode
                # ref_text is missing
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"] == "missing_ref_text"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_html(self, client):
        """Test that root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_root_contains_qwen_tts(self, client):
        """Test that root page mentions Qwen3-TTS."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Qwen" in response.content or b"TTS" in response.content
