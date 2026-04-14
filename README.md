# NanoTTS — C Inference Engine for MOSS-TTS-Nano

C implementation of [MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) voice cloning TTS.  
One `.dylib`/`.dll` with embedded model weights — no external model files needed.

> C + C++ (SentencePiece wrapper only)

## Performance

| Metric | Value |
|---|---|
| RTF (cold) | 0.70x |
| RTF (warm) | **0.33x** |
| vs PyTorch CPU | **1.8x faster** |
| vs naive C | **30x faster** |
| Output | 48kHz stereo |
| Model size | 308MB (bf16 + fp32) |

*Benchmarked on Apple M1, "Hello, nice to meet you. The weather is great today." with ljs.wav reference*

## API

```python
import ctypes

lib = ctypes.CDLL("./libnanotts.dylib")  # or nanotts.dll

lib.load_model()
lib.style_extract(b"ref.wav", b"speaker.codes")   # extract once, reuse
lib.generate_wav(b"speaker.codes", b"Hello!", ...)  # synthesize
lib.save_wav(b"output.wav", wav, samples, channels, sr)
lib.free_model()
```

See [example.py](example.py) for full usage.

## Build (macOS)

```bash
# Requires: OpenBLAS, SentencePiece, model.bin
make   # → libnanotts.dylib
```

## Optimizations Applied

From 68s → 2.3s (30x speedup):

| # | Optimization | Effect | Files |
|---|---|---|---|
| 1 | **NEON/SSE SIMD** (dot product, layernorm, vec_add) | 68s → 13s (5.2x) | ops.c |
| 2 | **Local transformer KV cache** (17 full passes → 17 single-token passes) | 13s → 8.5s (1.5x) | tts.c |
| 3 | **pthread attention head parallelism** | 8.5s → 7.0s (1.2x) | tts.c, codec.c |
| 4 | **pthread matmul row parallelism** | 7.0s → 4.2s (1.7x) | ops.c |
| 5 | **Workspace pre-allocation** (eliminate malloc/free per layer) | 4.2s → 3.7s (1.1x) | tts.c, codec.c |
| 6 | **Context-windowed attention** (skip out-of-window score computation) | 3.7s → 3.2s (1.2x) | codec.c |
| 7 | **OpenBLAS cblas_sgemm** | 3.2s → 2.3s (1.4x) | ops.c |
| 8 | **Fused FFN** (matmul+gelu+matmul in one pass) | ~5% | ops.c |
| 9 | **Weight pointer caching** (eliminate string lookup per layer) | ~3% | tts.c |
| 10 | **Codebook norm pre-computation** | ~2% | codec.c |
| 11 | **Resampling SIMD** (torchaudio-compatible sinc interpolation) | 30ms → 5ms | audio.c |
| 12 | **bf16 weight storage** (lossless, smaller file) | 531MB → 308MB | tensor.c |

## Cross-Platform Support

| Component | macOS | Windows |
|---|---|---|
| SIMD | NEON ✅ | SSE ✅ |
| Threading | pthread ✅ | CreateThread ✅ |
| Timer | clock_gettime ✅ | QueryPerformanceCounter ✅ |
| BLAS | OpenBLAS | OpenBLAS |
| SentencePiece | .dylib | .dll |

All platform branching is in `platform.h`.

## File Structure

```
nanotts.h           — Public API header
nanotts.c           — API implementation
tensor.h/c          — Tensor struct + binary weight loader
ops.h/c             — SIMD ops + OpenBLAS matmul + sampling
tts.h/c             — TTS model (GPT-2 transformer + generation loop)
codec.h/c           — Codec model (encoder + decoder + RLFQ quantizer)
prompt.h/c          — Prompt construction
sentencepiece.h/cpp — SentencePiece C++ wrapper (only C++ file)
audio.h/c           — WAV I/O + sinc resampling
platform.h          — OS abstraction (threading + timing)
embedded.h          — Binary embedding support
model.bin           — Model weights (not in repo, see Releases)
```
