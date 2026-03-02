# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Audio encoding service for TTS API.
Handles conversion of raw audio to various formats (mp3, opus, aac, flac, wav, pcm).
"""

import io
import logging
import struct
import asyncio
import shutil
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
COMPRESSED_FORMATS = {"mp3", "opus", "aac", "flac"}

# Default sample rate for Qwen3-TTS output
DEFAULT_SAMPLE_RATE = 24000


def get_content_type(audio_format: AudioFormat) -> str:
    """Get MIME content type for audio format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return content_types.get(audio_format, f"audio/{audio_format}")


def convert_to_wav(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to WAV format bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        WAV file bytes
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    if audio.size > 0:
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV header
    num_samples = len(audio_int16)
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')
    
    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # Data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio_int16.tobytes())
    
    return buffer.getvalue()


def convert_to_pcm(
    audio: np.ndarray,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to raw PCM bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        Raw PCM bytes
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    if audio.size > 0:
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


def encode_audio(
    audio: np.ndarray,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    Encode audio to the specified format.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Returns:
        Encoded audio bytes
    """
    if format == "wav":
        return convert_to_wav(audio, sample_rate)
    
    if format == "pcm":
        return convert_to_pcm(audio)
    
    # For compressed formats, use pydub if available, otherwise fall back to wav
    try:
        from pydub import AudioSegment
        
        # Convert to WAV first
        wav_bytes = convert_to_wav(audio, sample_rate)
        
        # Load into pydub
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        
        # Export to target format
        output = io.BytesIO()
        
        format_params = {
            "mp3": {"format": "mp3", "bitrate": "192k"},
            "opus": {"format": "opus", "bitrate": "128k"},
            "aac": {"format": "adts", "bitrate": "192k"},  # AAC in ADTS container
            "flac": {"format": "flac"},
        }
        
        params = format_params.get(format, {"format": format})
        export_format = params.pop("format", format)
        
        segment.export(output, format=export_format, **params)
        return output.getvalue()
        
    except ImportError:
        # Fall back to WAV if pydub not available
        logger.warning(f"pydub not available, returning WAV instead of {format}")
        return convert_to_wav(audio, sample_rate)
    except Exception as e:
        # Fall back to WAV on any encoding error
        logger.warning(f"Failed to encode to {format} ({e}), returning WAV")
        return convert_to_wav(audio, sample_rate)


def ensure_streaming_encoding_supported(format: AudioFormat) -> None:
    """Validate streaming encoding support for requested format."""
    if format in {"wav", "pcm"}:
        return

    if format not in COMPRESSED_FORMATS:
        raise ValueError(f"Unsupported response_format for streaming: {format}")

    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "Streaming compressed audio requires ffmpeg in PATH for response_format="
            f"{format}"
        )


def _build_streaming_ffmpeg_cmd(format: AudioFormat, sample_rate: int) -> list[str]:
    """Build ffmpeg command for incremental stdout encoding."""
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-flush_packets",
        "1",
        "-fflags",
        "+flush_packets",
    ]

    codec_args = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-write_xing", "0", "-f", "mp3"],
        "opus": ["-c:a", "libopus", "-b:a", "128k", "-application", "audio", "-f", "opus"],
        "aac": ["-c:a", "aac", "-b:a", "192k", "-f", "adts"],
        "flac": ["-c:a", "flac", "-f", "flac"],
    }

    if format not in codec_args:
        raise ValueError(f"Unsupported streaming ffmpeg format: {format}")

    return [*base, *codec_args[format], "pipe:1"]


class _StreamingFfmpegEncoder:
    """Stateful, incremental ffmpeg-based encoder for streaming compressed formats."""

    def __init__(self, format: AudioFormat, sample_rate: int):
        self.format = format
        self.sample_rate = sample_rate
        self._process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> None:
        cmd = _build_streaming_ffmpeg_cmd(self.format, self.sample_rate)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg binary not found while starting streaming encoder"
            ) from e

    async def _read_available_stdout(self, timeout_s: float = 0.01) -> bytes:
        if self._process is None or self._process.stdout is None:
            return b""

        out = bytearray()
        while True:
            try:
                chunk = await asyncio.wait_for(self._process.stdout.read(8192), timeout=timeout_s)
            except asyncio.TimeoutError:
                break

            if not chunk:
                break
            out.extend(chunk)

            if len(chunk) < 8192:
                break

        return bytes(out)

    async def encode_pcm_chunk(self, pcm_chunk: bytes) -> bytes:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Streaming encoder is not started")

        self._process.stdin.write(pcm_chunk)
        await self._process.stdin.drain()
        return await self._read_available_stdout()

    async def finish(self) -> bytes:
        if self._process is None:
            return b""

        if self._process.stdin is not None:
            self._process.stdin.close()
            await self._process.stdin.wait_closed()

        remaining = bytearray()
        if self._process.stdout is not None:
            while True:
                chunk = await self._process.stdout.read(8192)
                if not chunk:
                    break
                remaining.extend(chunk)

        stderr_bytes = b""
        if self._process.stderr is not None:
            stderr_bytes = await self._process.stderr.read()

        rc = await self._process.wait()
        if rc != 0:
            stderr_msg = stderr_bytes.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"ffmpeg streaming encoder exited with code {rc} for format={self.format}: {stderr_msg}"
            )

        return bytes(remaining)


async def encode_audio_streaming(
    audio_generator,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """
    Async generator that encodes audio chunks to the specified format.
    
    Args:
        audio_generator: Async generator yielding audio chunks as numpy arrays
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Yields:
        Encoded audio chunks
    """
    ensure_streaming_encoding_supported(format)

    stream_sample_rate = sample_rate
    wav_header_emitted = False
    ffmpeg_encoder: Optional[_StreamingFfmpegEncoder] = None

    try:
        async for item in audio_generator:
            if item is None:
                continue

            if isinstance(item, tuple):
                audio_chunk, chunk_sample_rate = item
                if chunk_sample_rate:
                    stream_sample_rate = int(chunk_sample_rate)
            else:
                audio_chunk = item

            if audio_chunk is None or len(audio_chunk) == 0:
                continue

            if format == "wav" and not wav_header_emitted:
                # Streaming WAV uses unknown-length RIFF/data sizes.
                num_channels = 1
                bits_per_sample = 16
                bytes_per_sample = bits_per_sample // 8
                byte_rate = stream_sample_rate * num_channels * bytes_per_sample
                block_align = num_channels * bytes_per_sample

                header = io.BytesIO()
                header.write(b"RIFF")
                header.write(struct.pack("<I", 0xFFFFFFFF))
                header.write(b"WAVE")
                header.write(b"fmt ")
                header.write(struct.pack("<I", 16))
                header.write(struct.pack("<H", 1))
                header.write(struct.pack("<H", num_channels))
                header.write(struct.pack("<I", stream_sample_rate))
                header.write(struct.pack("<I", byte_rate))
                header.write(struct.pack("<H", block_align))
                header.write(struct.pack("<H", bits_per_sample))
                header.write(b"data")
                header.write(struct.pack("<I", 0xFFFFFFFF))
                yield header.getvalue()
                wav_header_emitted = True

            pcm_chunk = convert_to_pcm(audio_chunk)

            if format in {"wav", "pcm"}:
                yield pcm_chunk
                continue

            if ffmpeg_encoder is None:
                ffmpeg_encoder = _StreamingFfmpegEncoder(format, stream_sample_rate)
                await ffmpeg_encoder.start()
            elif stream_sample_rate != ffmpeg_encoder.sample_rate:
                raise ValueError(
                    "Variable sample rate is not supported for compressed streaming response_format="
                    f"{format}: got {stream_sample_rate}, expected {ffmpeg_encoder.sample_rate}"
                )

            encoded_chunk = await ffmpeg_encoder.encode_pcm_chunk(pcm_chunk)
            if encoded_chunk:
                yield encoded_chunk

        if ffmpeg_encoder is not None:
            remaining = await ffmpeg_encoder.finish()
            if remaining:
                yield remaining
    finally:
        if ffmpeg_encoder is not None and ffmpeg_encoder._process is not None:
            if ffmpeg_encoder._process.returncode is None:
                ffmpeg_encoder._process.kill()
