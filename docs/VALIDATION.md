# 로컬 검증 기록 — 2026-09-20

검증 환경은 x86_64 Linux/WSL, Python 3.10.12, PyTorch 2.7.1+cpu, NumPy 1.26.4다. Jetson이나 OMAP 보드에 접속해 실행한 결과가 아니다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest -q
bash tools/prepare_jetson.sh --allow-cpu
bash tools/prepare_jetson.sh --install --allow-cpu
bash -n tools/prepare_jetson.sh
git diff --cached --check
```

43개 테스트가 통과했다. 테스트 범위는 실측 FIR과 DSP 일치, 잘못된 샘플레이트 거부, impulse gain·부호·지연, FFT wraparound 방지, 직접 convolution과 역전파 일치, block history, 신경망 인과성·출력 제한, 녹음 데이터의 단위·동기 길이·세션/PCM 누수 검사, 손상 WAV 거부, 녹음 준비에서 학습까지의 연결, checkpoint 저장·재개·다른 실험 덮어쓰기 방지, Jetson/CUDA/API 점검 판정이다. 테스트의 WAV는 합성 fixture이며 실제 실험 녹음이 아니다.

자동 준비 명령은 계수 파생 파일 생성과 실제 CPU optimizer 1 epoch 실행 후 `last.pt` 및 `best.pt` 저장까지 완료했다. `--install`에서는 기존 PyTorch를 유지하고 가상환경에 비-Torch 의존성만 설치했다. 이 PC에는 `ensurepip`가 없어 기존 pip를 공유하는 가상환경 생성 경로도 확인했다. 재실행마다 새로운 `runs/smoke/run-XXXXXXXX/` 폴더를 사용한다. 원본 RIR의 SHA-256은 계속 `9ece235a68a78a7f7c702d49d75525974df2ac2a8a9ca42ced155af901bcf9e3`이다.

500탭, float32, batch 1, 입력 2,640 samples, CPU thread 2에서 `torch.utils.benchmark.Timer`로 FIR 순전파와 역전파의 중앙값을 비교했다. 직접 FIR은 약 0.933 ms, FFT 선형 convolution은 약 0.171 ms였다. 이는 이 개발 PC의 해당 크기에서 측정한 약 5.45배 차이이며, Jetson 속도·전체 신경망 지연·실시간 지연 보장으로 해석하지 않는다.

소스 그대로 보존한 firmware에는 기존 CRLF/공백을 허용했다. 기존 GCRN `scripts/`의 코드 내용은 변경하지 않았다. 실제 Jetson CUDA 실행, CCS rebuild, ANC-OFF 실측 녹음 학습, 실시간 장치 통신과 음향 감쇠는 다음 환경에서 확인할 항목으로 남는다.
