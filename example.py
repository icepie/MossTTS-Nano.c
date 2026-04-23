"""End-to-end TTS inference example using ctypes."""
import ctypes
import os
import platform

# ===== Setup =====
if platform.system() == "Linux":
    lib_path = "./libnanotts.so"
elif platform.system() == "Darwin":
    lib_path = "./libnanotts.dylib"
else:
    lib_path = "./moss_tts.dll"

ref_wav = os.environ.get(
    "NANOTTS_REF_WAV",
    "/data/Projects/asr_server/test/asr/test_wavs/en.wav",
)

lib = ctypes.CDLL(lib_path)

lib.load_model.restype = ctypes.c_int
lib.free_model.restype = None
lib.style_extract.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.style_extract.restype = ctypes.c_int
lib.generate_wav.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.generate_wav.restype = ctypes.c_int
lib.generate_wav_from_ref.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.generate_wav_from_ref.restype = ctypes.c_int
lib.save_wav.argtypes = [
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.save_wav.restype = ctypes.c_int

# ===== 모델 로드 =====
lib.load_model()

# ===== 방법 1: 레퍼런스 wav에서 바로 합성 =====
wav = ctypes.POINTER(ctypes.c_float)()
samples = ctypes.c_int()
channels = ctypes.c_int()
sr = ctypes.c_int()

lib.generate_wav_from_ref(
    ref_wav.encode(),                     # 레퍼런스 음성
    "Hello, nice to meet you.".encode(),  # 합성할 텍스트
    ctypes.byref(wav), ctypes.byref(samples),
    ctypes.byref(channels), ctypes.byref(sr),
    1,  # 1=stereo, 0=mono
)
lib.save_wav(b"output_direct.wav", wav, samples, channels, sr)
lib.free(wav)
print(f"Direct: {samples.value} samples, {channels.value}ch, {sr.value}Hz")

# ===== 방법 2: 스타일 캐시 사용 =====
# 스타일 추출 (1번만)
lib.style_extract(ref_wav.encode(), b"ljs.codes")

# 캐시된 스타일로 여러 문장 합성
texts = [
    b"The weather is beautiful today.",
    b"\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94.",  # 안녕하세요
    b"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf.",  # こんにちは
]

for i, text in enumerate(texts):
    wav = ctypes.POINTER(ctypes.c_float)()
    samples = ctypes.c_int()
    channels = ctypes.c_int()
    sr = ctypes.c_int()

    lib.generate_wav(
        b"ljs.codes", text,
        ctypes.byref(wav), ctypes.byref(samples),
        ctypes.byref(channels), ctypes.byref(sr),
        1,
    )
    filename = f"output_cached_{i}.wav".encode()
    lib.save_wav(filename, wav, samples, channels, sr)
    lib.free(wav)
    print(f"Cached {i}: {samples.value} samples")

# ===== 정리 =====
lib.free_model()
print("Done!")
