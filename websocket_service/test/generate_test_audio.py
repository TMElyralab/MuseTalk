#!/usr/bin/env python3
"""
Generate test audio files for MuseTalk WebSocket testing.
"""
import numpy as np
import wave
import struct
import os


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave_data = np.sin(2 * np.pi * frequency * t)
    return wave_data


def generate_speech_like_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate audio that resembles speech patterns."""
    samples = int(sample_rate * duration)
    audio = np.zeros(samples)
    
    # Mix multiple frequencies to simulate speech
    # Fundamental frequency (voice pitch)
    f0 = 120 + np.random.randn() * 20  # 100-140 Hz
    
    # Add harmonics
    for harmonic in range(1, 5):
        freq = f0 * harmonic
        amplitude = 1.0 / harmonic
        t = np.linspace(0, duration, samples)
        audio += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add formants (speech characteristics)
    formants = [700, 1220, 2600]  # Typical formant frequencies
    for formant in formants:
        t = np.linspace(0, duration, samples)
        audio += 0.3 * np.sin(2 * np.pi * formant * t)
    
    # Add some noise
    audio += 0.05 * np.random.randn(samples)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio


def save_wav(filename: str, audio_data: np.ndarray, sample_rate: int = 16000):
    """Save audio data as WAV file."""
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        # Set parameters
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Write data
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"Saved: {filename}")


def save_pcm(filename: str, audio_data: np.ndarray):
    """Save audio data as raw PCM file."""
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    with open(filename, 'wb') as f:
        f.write(audio_int16.tobytes())
    
    print(f"Saved: {filename}")


def generate_test_chunks(duration: float = 1.0, chunk_duration_ms: int = 40) -> list:
    """Generate audio chunks for streaming."""
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    
    # Generate full audio
    audio = generate_speech_like_audio(duration, sample_rate)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) == chunk_samples:
            chunks.append(chunk)
    
    return chunks


def main():
    """Generate various test audio files."""
    # Create fixtures directory
    os.makedirs("fixtures", exist_ok=True)
    
    # 1. Simple sine wave (1 second)
    audio = generate_sine_wave(440, 1.0)  # A4 note
    save_wav("fixtures/sine_440hz_1s.wav", audio)
    
    # 2. Speech-like audio (2 seconds)
    audio = generate_speech_like_audio(2.0)
    save_wav("fixtures/speech_like_2s.wav", audio)
    save_pcm("fixtures/speech_like_2s.pcm", audio)
    
    # 3. Short audio for testing (40ms - one chunk)
    audio = generate_speech_like_audio(0.04)
    save_pcm("fixtures/single_chunk_40ms.pcm", audio)
    
    # 4. Generate multiple 40ms chunks
    chunks = generate_test_chunks(1.0, 40)
    print(f"\nGenerated {len(chunks)} chunks of 40ms audio")
    
    # Save first few chunks
    for i, chunk in enumerate(chunks[:5]):
        save_pcm(f"fixtures/chunk_{i:03d}_40ms.pcm", chunk)
    
    # 5. Create a test config
    import json
    config = {
        "test_user_id": "test_user",
        "sample_rate": 16000,
        "chunk_duration_ms": 40,
        "chunks_per_second": 25,
        "audio_files": {
            "sine": "sine_440hz_1s.wav",
            "speech": "speech_like_2s.wav",
            "chunks": [f"chunk_{i:03d}_40ms.pcm" for i in range(5)]
        }
    }
    
    with open("fixtures/test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nTest audio files generated successfully!")
    print("You can use these files with the test client.")


if __name__ == "__main__":
    main()