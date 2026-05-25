"""공유 E5 임베더 싱글톤.

NERFactChecker와 ConsistencyChecker가 같은 모델을 각자 로드해
메모리를 두 배 소비하는 문제를 방지한다.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("lawsguard.embedder")

_lock = threading.Lock()
_embedder = None


def get_embedder():
    """intfloat/multilingual-e5-large 싱글톤을 반환한다. 실패 시 None."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is not None:
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            logger.warning("sentence_transformers 미설치 — 시맨틱 임베딩 비활성화")
            return None
        try:
            from ..config import config
            model_name = config.rag.embedding_model
        except Exception:
            model_name = "intfloat/multilingual-e5-large"
        try:
            device = _get_device()
            _embedder = SentenceTransformer(model_name, device=device, local_files_only=True)
            logger.info("E5 임베더 로드 완료: %s (%s)", model_name, device)
        except Exception as e:
            logger.warning("E5 임베더 로드 실패: %s", e)
            _embedder = None
    return _embedder


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
