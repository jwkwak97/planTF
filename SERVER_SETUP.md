# PlanTF B200 (Blackwell) Server Setup Guide

> Target: NVIDIA B200 (Blackwell, sm_100) | CUDA 12.8+ | PyTorch 2.10+ cu128

---

## NATTEN Blackwell 지원 여부 (핵심)

**결론: NATTEN 0.21.x부터 Blackwell (sm_100) 네이티브 커널 완전 지원.**

- NATTEN 0.21.6 (2026-04-14) 기준 sm_100 / sm_103 FNA/FMHA 커널 탑재
- 단, **PyTorch 2.10+ (cu128 이상)** 에서만 Blackwell 커널 활성화
- PyTorch 2.9 이하에서는 Blackwell 커널 미포함 → 반드시 2.10 이상 사용

---

## 0. 시스템 요구사항 확인

```bash
# NVIDIA 드라이버 버전 확인 (R570 이상 필요)
nvidia-smi

# CUDA Toolkit 버전 확인 (12.8 이상 필요)
nvcc --version

# GPU 아키텍처 확인 (B200이면 sm_100 출력)
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# 시스템 RAM 확인 (컴파일 시 대용량 필요)
free -h
```

---

## 1. Conda 환경 생성

```bash
conda create -n plantf python=3.10 -y
conda activate plantf
```

---

## 2. PyTorch 설치 (Blackwell 지원 버전)

```bash
# CUDA 12.8 기반 (cu128) — B200 권장
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability())"
# 예상 출력: 2.x.x+cu128, True, (10, 0)
```

---

## 3. nuplan-devkit 설치

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
cd ..
```

---

## 4. PlanTF 설치

```bash
pip install -r requirements.txt
sh ./script/setup_env.sh
```

---

## 5. NATTEN 설치 (Blackwell sm_100)

> **주의:** 소스 컴파일 시 시스템 RAM 고갈로 서버 접속 불가 현상이 발생할 수 있습니다.
> Pre-built wheel 사용을 우선 시도하세요.

**방법 1 (권장): Pre-built wheel — 컴파일 없음**

```bash
# natten.org/install 에서 본인 환경(Python / PyTorch / CUDA)에 맞는 URL 확인
pip install natten -f https://shi-labs.com/natten/wheels/cu128/torch2.x/index.html
```

**방법 2: 소스 컴파일 (RAM 제한 필수)**

```bash
# MAX_JOBS=2로 병렬 컴파일 제한하여 OOM 방지
MAX_JOBS=2 NATTEN_CUDA_ARCH="10.0" pip install natten==0.21.6
```

설치 확인:

```bash
python -c "import natten; print(natten.__version__); natten.check_device()"
```

---

## 6. 추가 의존성

```bash
pip install numba tensorboard wandb rich timm
pip install pytorch-lightning==2.0.1 torchmetrics==0.10.2
```

---

## 7. 데이터셋 환경변수 설정

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/maps"
export NUPLAN_EXP_ROOT="/path/to/experiment/output"
```

`~/.bashrc` 또는 `~/.bash_profile`에 추가 권장.

---

## 8. 학습 실행 예시

```bash
# 단일 GPU
python run_training.py \
    experiment_name=planTF_test \
    py_func=train \
    +training=training_planTF

# 멀티 GPU (DDP)
python run_training.py \
    experiment_name=planTF_test \
    py_func=train \
    +training=training_planTF \
    worker=sequential \
    scenario_builder.num_workers=40
```

---

## 참고: NATTEN CUDA 아키텍처 코드표

| GPU         | Architecture    | sm 코드 | NATTEN_CUDA_ARCH |
|-------------|-----------------|---------|------------------|
| B200        | Blackwell       | sm_100  | `10.0`           |
| B300        | Blackwell Ultra | sm_103  | `10.3`           |
| H100 / H200 | Hopper          | sm_90   | `9.0`            |
| A100        | Ampere          | sm_80   | `8.0`            |

---

## 트러블슈팅

### [실제 경험] NATTEN 설치 후 서버 접속 불가 + OOM

**증상:** `pip install natten==0.21.6` 실행 후 서버 응답 없어지고 OOM 발생

**원인:** NATTEN은 pip install 시 수천 개의 CUDA 커널을 소스 컴파일합니다.
Blackwell (sm_100) 커널은 특히 무거워서 병렬 컴파일이 시스템 RAM을 전부 소진하고,
OOM killer가 SSH 데몬까지 종료시켜 서버 접속이 불가해집니다.
GPU OOM이 아니라 **CPU/System RAM OOM**이 원인입니다.

**해결:** Pre-built wheel 사용 또는 `MAX_JOBS=2` 제한 컴파일 (5번 항목 참조)

---

### NATTEN 컴파일 중 CUDA 경로 오류

```bash
export CUDA_HOME=/usr/local/cuda-12.8
MAX_JOBS=2 NATTEN_CUDA_ARCH="10.0" pip install natten==0.21.6
```

### PyTorch가 GPU를 인식하지 못할 때

```bash
# 드라이버 버전이 R570 미만인 경우 업그레이드 필요
sudo apt install --only-upgrade nvidia-driver-570
```

### `nvcc: command not found`

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
