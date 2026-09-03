# Deep ANC 실시간 추론 데모 이미지 — Jetson AGX Orin, engine.type: ort (ONNX Runtime, CPU).
# GPU/torch 불필요 (학습 전용 스택은 포함하지 않는다) — A/Space 키로 ANC ON/OFF 토글.
#
# 빌드: docker build -t deep-anc-runtime .
# 실행 (오디오 장치 + 실시간 스케줄링 권한 필요):
#   docker run -it --rm \
#     --device /dev/snd --group-add audio \
#     --cap-add=sys_nice --ulimit rtprio=95 \
#     deep-anc-runtime \
#     --config configs/runtime_tiny_acoustic_pilot_scratch.yaml \
#     --confirm-speaker --confirm-user-present --confirm-volume-minimum
#
# 시작 후: A 또는 Space = ANC ON/OFF, N = 소음 ON/OFF, Q = 종료.
# 주의: 기본 진입점은 run_realtime.py의 표준 안전 게이트(assert_measurement_preconditions,
# input_preflight)를 그대로 거친다 — 우회하지 않는다. 이 세션에서 확인된 REF/ERR 마이크의
# 잔여 DC bias/광대역 잡음floor 이슈가 남아 있으면 이 게이트가 정상적으로 거부할 수 있다.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-jetson.txt ./
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.15.3 \
    PyYAML==5.4.1 \
    soundfile==0.14.0 \
    sounddevice==0.5.5 \
    onnx>=1.15 \
    onnxruntime==1.18.1

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/demo/ ./scripts/demo/
COPY runs/export/ ./runs/export/

ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "-m", "deep_anc.realtime.run_realtime"]
