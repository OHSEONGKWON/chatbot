# GPU 사용 가이드

모든 평가 스크립트가 **자동으로 GPU를 감지하고 사용**합니다.

## GPU 사용 현황

### ✅ GPU 자동 사용
다음 작업들이 GPU에서 실행됩니다:

1. **Dense Retrieval (BGE-M3)**
   - 문서 임베딩 계산
   - 쿼리 임베딩 계산
   - 코사인 유사도 계산

2. **최적화 기능**
   - FP16 (Mixed Precision) 자동 활성화
   - 배치 크기 자동 조정
   - 메모리 자동 관리

**참고:** NER 모델은 메인 파이프라인에서 이미 실행되므로, 평가 스크립트에서는 별도로 실행하지 않습니다.

### ❌ GPU 불필요
다음 작업들은 CPU에서 충분합니다:

- BM25 검색 (순수 Python 연산)
- OpenAI API 호출 (외부 API)
- 결과 분석 및 시각화

---

## GPU 요구사항

### 최소 사양
- **VRAM**: 4GB 이상
- **CUDA**: 11.0 이상
- **PyTorch**: CUDA 지원 버전

### 권장 사양
- **VRAM**: 8GB 이상 (RTX 3060, RTX 4060 등)
- **CUDA**: 11.8 이상
- **PyTorch**: 최신 CUDA 버전

---

## GPU 확인 및 설정

### 1. CUDA 설치 확인

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA 버전: {torch.version.cuda}')"
```

### 2. PyTorch CUDA 버전 설치

```bash
# CUDA 11.8 기준 (nvidia-smi로 확인한 CUDA 버전에 맞춰 설치)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. GPU 정보 확인

스크립트 실행 시 자동으로 출력됩니다:

```
================================================================================
GPU 정보
================================================================================
✓ CUDA 사용 가능
  - GPU 개수: 1
  - GPU 0: NVIDIA GeForce RTX 3060
    • 할당된 메모리: 0.00 GB
    • 예약된 메모리: 0.00 GB
    • 전체 메모리: 12.00 GB
  - CUDA 버전: 11.8
  - PyTorch 버전: 2.0.1+cu118
================================================================================
```

---

## 메모리 최적화

### FP16 (Mixed Precision)

**자동 활성화**: GPU 사용 시 기본적으로 활성화됩니다.

**효과:**
- VRAM 사용량 50% 감소
- 연산 속도 2-3배 향상
- 정확도 영향 거의 없음

**수동 비활성화** (필요시):
```python
# compare_retrieval.py에서
dense = DenseRetriever(use_fp16=False)
```

### 배치 크기 자동 조정

스크립트가 GPU VRAM에 따라 배치 크기를 자동으로 조정합니다:

| VRAM | 배치 크기 |
|------|----------|
| < 4GB | 8 |
| 4-8GB | 16 |
| 8-16GB | 32 |
| > 16GB | 32 |

**수동 조정** (필요시):
```python
# compare_retrieval.py에서
self.batch_size = 16  # 원하는 크기로 변경
```

### 메모리 부족 해결

#### 1. 배치 크기 줄이기
```python
# compare_retrieval.py의 DenseRetriever.__init__에서
self.batch_size = 8  # 또는 더 작게
```

#### 2. FP16 활성화 확인
```python
# 이미 기본 활성화되어 있지만, 명시적으로:
dense = DenseRetriever(use_fp16=True)
```

#### 3. GPU 캐시 정리
```python
from gpu_utils import clear_gpu_cache
clear_gpu_cache()
```

#### 4. CPU 모드로 강제 실행
```bash
# 환경변수 설정
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
set CUDA_VISIBLE_DEVICES=       # Windows CMD
$env:CUDA_VISIBLE_DEVICES=""    # Windows PowerShell

# 그 후 스크립트 실행
python scripts/compare_retrieval.py
```

---

## 성능 비교

### 문서 임베딩 (5000개 문서 기준)

| 환경 | 시간 | 메모리 |
|------|------|--------|
| **CPU** | ~45분 | 4GB |
| **GPU (FP32)** | ~5분 | 8GB VRAM |
| **GPU (FP16)** | ~3분 | 4GB VRAM |

### 전체 평가 (200 케이스 기준)

| Phase | CPU | GPU (FP16) | 개선 |
|-------|-----|------------|------|
| Phase 1 | 60분 | 20분 | **3x** |
| Phase 2 | 180분 | 90분 | **2x** |
| 합계 | 240분 | 110분 | **2.2x** |

---

## 문제 해결

### CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**해결:**
1. 배치 크기 줄이기 (위 참조)
2. FP16 활성화 확인
3. 다른 GPU 프로세스 종료
4. GPU 캐시 정리

### GPU 미감지

**증상:**
```
⚠️  CUDA 사용 불가 - CPU 모드로 실행됩니다
```

**해결:**
1. `nvidia-smi` 실행하여 GPU 확인
2. CUDA 버전 확인
3. PyTorch CUDA 버전 재설치:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 속도가 느림

**원인:**
- CPU 모드로 실행 중
- 배치 크기가 너무 작음
- FP16 비활성화

**확인:**
스크립트 시작 시 출력되는 GPU 정보 확인:
```
디바이스: cuda
FP16 (Mixed Precision) 활성화
배치 크기: 32
```

---

## 모니터링

### GPU 사용률 실시간 모니터링

```bash
# 터미널 1: 스크립트 실행
python scripts/compare_retrieval.py

# 터미널 2: GPU 모니터링
watch -n 1 nvidia-smi  # Linux/Mac

# Windows PowerShell
while($true) { nvidia-smi; sleep 1; cls }
```

### 메모리 사용량 확인

스크립트 실행 중 자동으로 출력됩니다:
```
시작 전 GPU 0 메모리: 0.00 GB / 0.00 GB (할당/예약)
완료 후 GPU 0 메모리: 2.34 GB / 3.50 GB (할당/예약)
```

---

## 참고

- **PyTorch CUDA 가이드**: https://pytorch.org/get-started/locally/
- **NVIDIA CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **Mixed Precision Training**: https://pytorch.org/docs/stable/amp.html
