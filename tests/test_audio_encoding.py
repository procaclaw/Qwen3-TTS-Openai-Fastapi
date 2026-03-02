# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for audio encoding helpers.
"""

import numpy as np
import pytest

from api.services import audio_encoding


@pytest.mark.asyncio
async def test_encode_audio_streaming_wav_emits_header_then_pcm():
    async def fake_audio_stream():
        yield np.array([0.0, 0.5], dtype=np.float32), 24000

    chunks = []
    async for chunk in audio_encoding.encode_audio_streaming(
        fake_audio_stream(), format="wav", sample_rate=24000
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].startswith(b"RIFF")
    assert len(chunks[1]) > 0


@pytest.mark.asyncio
async def test_encode_audio_streaming_pcm_emits_raw_pcm_only():
    async def fake_audio_stream():
        yield np.array([0.0, 0.5], dtype=np.float32), 24000

    chunks = []
    async for chunk in audio_encoding.encode_audio_streaming(
        fake_audio_stream(), format="pcm", sample_rate=24000
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == audio_encoding.convert_to_pcm(np.array([0.0, 0.5], dtype=np.float32))


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac"])
async def test_encode_audio_streaming_compressed_uses_incremental_encoder(monkeypatch, fmt):
    events = []

    class FakeStreamingEncoder:
        def __init__(self, format, sample_rate):
            self.format = format
            self.sample_rate = sample_rate
            self._process = None
            self._count = 0

        async def start(self):
            events.append(("start", self.format, self.sample_rate))

        async def encode_pcm_chunk(self, pcm_chunk: bytes) -> bytes:
            self._count += 1
            events.append(("chunk", self._count, len(pcm_chunk)))
            return f"{self.format}-{self._count}".encode("utf-8")

        async def finish(self) -> bytes:
            events.append(("finish", self._count))
            return b"tail"

    monkeypatch.setattr(audio_encoding, "_StreamingFfmpegEncoder", FakeStreamingEncoder)
    monkeypatch.setattr(audio_encoding, "ensure_streaming_encoding_supported", lambda _fmt: None)

    async def fake_audio_stream():
        yield np.array([0.0, 0.1], dtype=np.float32), 24000
        yield np.array([0.2, 0.3], dtype=np.float32), 24000

    out = []
    async for chunk in audio_encoding.encode_audio_streaming(
        fake_audio_stream(), format=fmt, sample_rate=24000
    ):
        out.append(chunk)

    assert out == [f"{fmt}-1".encode("utf-8"), f"{fmt}-2".encode("utf-8"), b"tail"]
    assert events[0] == ("start", fmt, 24000)
    assert events[1][0] == "chunk"
    assert events[2][0] == "chunk"
    assert events[3] == ("finish", 2)
