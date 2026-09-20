# 기존 GCRN Speech Enhancement 실험

이 문서는 이전 저장소 README를 보존한 것이다. 아래 Docker image, 경로, 환경은 과거 데스크톱 실험 기록이며 Jetson Orin 설치 절차가 아니다. 현재 ANC 준비는 [프로젝트 README](../README.md)와 [Jetson 안내](JETSON_SETUP.md)를 따른다. `scripts/train.py`의 `mix/sph` 학습은 측정 2차경로를 사용하는 ANC 학습과 별개다.

GCRN(Complex-valued Gated Convolutional Recurrent Network) 기반 Speech Enhancement 및 향후 Deep ANC(Active Noise Cancellation) 연구를 위한 실험 저장소입니다.

본 프로젝트는 Docker 환경에서 GCRN을 학습하고, LibriSpeech와 DNS-Challenge 데이터셋을 이용하여 noisy-clean speech pair를 생성한 뒤 Speech Enhancement 성능을 평가하는 것을 목표로 합니다.

---

# Docker Image

DockerHub

```bash
docker pull jeongsj/deepanc:speech
```

DockerHub Repository

```text
jeongsj/deepanc:speech
```

---

# Repository Clone

```bash
git clone https://github.com/Roka-jsj/DeepANC.git
cd DeepANC
```

---

# Run Docker

```bash
docker run --gpus all -it \
  --name deepanc \
  -v $(pwd):/workspace/DeepANC \
  -v ~/DeepANC/datasets:/workspace/datasets \
  jeongsj/deepanc:speech bash
```

GPU 확인

```bash
nvidia-smi

python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

---

# Experimental Environment

## Hardware

* GPU: NVIDIA RTX 3080 Ti
* RAM: 64 GB

## Software

* Ubuntu 22.04
* Docker
* CUDA 11.1
* PyTorch 1.9.0

---

# Dataset

## Clean Speech

### LibriSpeech

Official Website:

https://www.openslr.org/12

Downloaded Dataset:

```text
train-clean-100
```

Number of files:

```text
28,539 FLAC files
```

Total duration:

```text
100 hours
```

Download:

```bash
mkdir -p ~/DeepANC/datasets/LibriSpeech
cd ~/DeepANC/datasets/LibriSpeech

wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

tar -xzf train-clean-100.tar.gz
```

---

## Noise Dataset

### DNS-Challenge

Repository:

https://github.com/microsoft/DNS-Challenge

Used Dataset:

```text
noise_fullband
```

Download:

```bash
mkdir -p ~/DeepANC/datasets
cd ~/DeepANC/datasets

git clone https://github.com/microsoft/DNS-Challenge.git

cd DNS-Challenge

curl -L "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2" \
| tar -C "./" -f - -x -j
```

---

# Data Generation

## Preprocessing

* Convert to 16 kHz
* Convert to Mono
* RMS Normalization

## Noisy-Clean Pair Generation

```text
Clean Speech
+
Noise
↓
Noisy Speech
```

Random SNR:

```text
-5 dB ~ 15 dB
```

Generated Dataset:

```text
Total      : 10000 pairs
Train Set  : 9000 pairs
Valid Set  : 1000 pairs
```

---

# GCRN Data Format

GCRN 학습 데이터는 HDF5 `.ex` 형식을 사용합니다.

```text
sample.ex
├── mix
└── sph
```

Description

```text
mix : Noisy Speech
sph : Clean Speech
```

---

# Training Configuration

```text
batch_size      = 4
buffer_size     = 8
learning_rate   = 0.0005
max_n_epochs    = 30
clip_norm       = 5.0

segment_size    = 4 sec
segment_shift   = 1 sec
```

---

# Training

```bash
cd /workspace/DeepANC/scripts

python -B ./train.py \
  --gpu_ids=0 \
  --tr_list=../filelists/tr_list_librispeech.txt \
  --cv_file=../data/datasets/cv_librispeech/cv_librispeech.ex \
  --ckpt_dir=exp_librispeech_dns \
  --logging_period=100 \
  --clip_norm=5.0 \
  --lr=0.0005 \
  --time_log=./time.log \
  --unit=utt \
  --batch_size=4 \
  --buffer_size=8 \
  --max_n_epochs=30
```

---

# Evaluation

## Test

```bash
python -B ./test.py \
  --gpu_ids=0 \
  --tt_list=../filelists/tt_list_librispeech.txt \
  --ckpt_dir=exp_librispeech_dns \
  --model_file=./exp_librispeech_dns/models/latest.pt
```

---

## Metrics

### SNR

```bash
python -B ./measure.py \
  --metric=snr \
  --tt_list=../filelists/tt_list_librispeech.txt \
  --ckpt_dir=exp_librispeech_dns
```

### STOI

```bash
python -B ./measure.py \
  --metric=stoi \
  --tt_list=../filelists/tt_list_librispeech.txt \
  --ckpt_dir=exp_librispeech_dns
```

### PESQ

```bash
python -B ./measure.py \
  --metric=pesq \
  --tt_list=../filelists/tt_list_librispeech.txt \
  --ckpt_dir=exp_librispeech_dns
```

---

# Experimental Results

## Dataset

```text
Clean Speech : LibriSpeech train-clean-100
Noise        : DNS-Challenge noise_fullband
```

## Training

```text
Train Pairs      : 9000
Validation Pairs : 1000

Epoch            : 30
Learning Rate    : 0.0005
```

## SNR Result

Input SNR:

```text
6.0834 dB
```

Output SNR:

```text
8.9397 dB
```

Improvement:

```text
+2.8562 dB
```

---

# Output Files

```text
*_mix.wav
```

Noisy input

```text
*_sph.wav
```

Clean target

```text
*_sph_est.wav
```

Enhanced output

---

# Project Structure

```text
DeepANC
├── scripts
│   ├── train.py
│   ├── test.py
│   ├── measure.py
│   ├── run_train.sh
│   ├── run_evaluate.sh
│   └── utils
│
├── README.md
└── .gitignore
```

---

# Notes

본 저장소에는 대용량 데이터셋과 모델 파일(.pt)을 포함하지 않습니다.

구성:

```text
DockerHub    : 실행 환경
GitHub       : 코드 및 문서
Datasets     : LibriSpeech / DNS-Challenge
Checkpoints  : 학습된 모델(.pt)
```

---

# Future Work

* DNS Full Dataset
* LibriSpeech Full Dataset
* Streaming Inference
* Jetson AGX Orin Deployment
* Deep ANC
* End-to-End Noise Cancellation
