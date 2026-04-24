import os
import json
import re
from dataclasses import dataclass

import torch
from transformers import pipeline

from modules.corrector import AnswerCorrector


@dataclass
class NERCheckResult:
    original_answer: str
    corrected_answer: str
    found_entities: list
    mismatched_entities: list[dict]
    was_corrected: bool

class NERFactChecker:
    def __init__(self, model_path=None):
        print("[System] NER FactChecker (Context-Anchored) 로딩 중...")
        resolved_model_path = self._resolve_model_path(model_path)
        device = 0 if torch.cuda.is_available() else -1
        print(f"[System] NER model={resolved_model_path}, device={'cuda:0' if device == 0 else 'cpu'}")
        self.ner_pipeline = pipeline(
            "ner",
            model=resolved_model_path,
            tokenizer=resolved_model_path,
            aggregation_strategy="simple",
            device=device,
        )
        self.target_labels = ['LAW', 'PENALTY', 'AMOUNT', 'DATE', 'ORG', 'CRIME']
        self.domain_keywords = {
            "sexual": ["성추행", "강제추행", "성희롱", "성폭력", "강간", "디지털성범죄", "성범죄", "성착취"],
            "labor": ["근로", "해고", "임금", "노동", "최저임금", "취업규칙", "근로기준"],
            "finance": ["사기", "대출", "채권", "금융", "이자", "변제", "채무"],
        }
        self._corrector = AnswerCorrector()

    def _resolve_model_path(self, model_path):
        env_model_path = os.getenv("LAWSGUARD_NER_MODEL")
        local_candidates = [
            "outputs/legal-ner-lawsguard-v1-gpu",
            "outputs/legal-ner-lawsguard-v1-fastfull",
            "outputs/legal-ner-lawsguard-v1",
            "outputs/legal-ner-klue",
            "outputs/legal-ner-smoke",
        ]
        candidate_paths = [
            env_model_path,
            *local_candidates,
            model_path,
            "klue/roberta-base",
        ]

        for path in candidate_paths:
            if not path:
                continue
            if path == "klue/roberta-base":
                return path
            if os.path.isdir(path):
                if self._is_valid_local_model_dir(path):
                    return path
                print(f"[System] skip invalid checkpoint: {path}")
                continue
            if path in local_candidates:
                continue
            # 허브 모델 id 같은 비-디렉토리 문자열은 그대로 시도
            if isinstance(path, str) and path.strip():
                return path

        return "klue/roberta-base"

    def _is_valid_local_model_dir(self, model_dir):
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.isfile(config_path):
            return False
        if not os.path.isfile(model_path):
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return bool(cfg.get("model_type"))
        except Exception:
            return False

    def extract_entities(self, text):
        """텍스트에서 지정된 6대 법률 개체명을 추출합니다."""
        entities = self.ner_pipeline(text)
        normalized = []
        for ent in entities:
            if ent.get("entity_group") not in self.target_labels:
                continue

            start = ent.get("start")
            end = ent.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                surface = text[start:end]
            else:
                surface = ent.get("word", "")

            surface = surface.strip()
            if not surface:
                continue
            if len(surface) == 1:
                continue

            cloned = dict(ent)
            cloned["word"] = surface
            normalized.append(cloned)

        return normalized

    def find_hallucinations(self, llm_answer, rag_chunks):
        """
        LLM 답변의 환각을 RAG 원문과 대조하여 탐지합니다.
        (법령명 기반 컨텍스트 앵커링 적용)
        """
        # 1. LLM 답변에서 개체명 추출 (문장 등장 순서대로 정렬됨)
        ans_entities = self.extract_entities(llm_answer)
        hallucinations = []
        
        # 전체 RAG 텍스트 (Fallback 및 LAW 자체 검증용)
        full_rag_text = " ".join([chunk.get("text", "") for chunk in rag_chunks])
        answer_domains = self._detect_domains(llm_answer)
        rag_domains = self._detect_domains(full_rag_text)
        domain_aligned = self._is_domain_aligned(answer_domains, rag_domains)
        fallback_law_anchors = self._extract_law_anchors_from_text(llm_answer)
        
        # 현재 검사 중인 기준 법령(Anchor)
        current_context_law = fallback_law_anchors[0] if fallback_law_anchors else None

        for ans_ent in ans_entities:
            label = ans_ent['entity_group']
            word = ans_ent['word']
            start = ans_ent.get("start")
            end = ans_ent.get("end")

            # 💡 [핵심 로직 1] 법령(LAW)이 등장하면 기준 컨텍스트(Anchor)를 업데이트
            if label == 'LAW':
                if not self._is_reliable_law_anchor(word):
                    hallucinations.append({
                        "label": label,
                        "wrong_word": word,
                        "correct_word": None,
                        "context_law": "N/A",
                        "start": start,
                        "end": end,
                        "reason_code": "LAW_ANCHOR_UNRELIABLE",
                        "reason": "법령 앵커가 숫자 조각/불완전 토큰으로 판단되어 컨텍스트 앵커로 사용하지 않았습니다.",
                    })
                    continue

                current_context_law = word
                
                # 법령 이름 자체가 RAG에 없는 환각인지 1차 검사
                # (텍스트 본문이나 메타데이터에 해당 법령 이름이 없으면 지어낸 법령으로 간주)
                is_law_valid = (word in full_rag_text) or any(
                    word in str(v) for chunk in rag_chunks for v in chunk.get("metadata", {}).values()
                )
                
                if not is_law_valid:
                    hallucinations.append({
                        "label": label,
                        "wrong_word": word,
                        "correct_word": None, # 지어낸 법이라 교정 후보 찾기 어려움
                        "context_law": "N/A",
                        "start": start,
                        "end": end,
                        "reason_code": "LAW_NOT_FOUND_IN_RAG",
                        "reason": "답변에서 언급된 법령명이 RAG 문서/메타데이터에서 확인되지 않아 교정을 차단했습니다.",
                    })
                continue # LAW 자체는 컨텍스트만 업데이트하고 다음 개체로 넘어감

            # 💡 [게이트 1] 질문/답변 도메인과 RAG 도메인이 맞지 않으면 교정을 차단
            if not domain_aligned:
                if word not in full_rag_text:
                    hallucinations.append({
                        "label": label,
                        "wrong_word": word,
                        "correct_word": None,
                        "context_law": current_context_law,
                        "start": start,
                        "end": end,
                        "reason_code": "DOMAIN_MISMATCH_BLOCKED",
                        "reason": f"답변 도메인({sorted(answer_domains)})과 RAG 도메인({sorted(rag_domains)})이 불일치하여 자동 교정을 차단했습니다.",
                    })
                continue

            # 💡 [핵심 로직 2] 컨텍스트 앵커링 기반 타겟 RAG 구역 설정
            target_rag_text = full_rag_text # 기본값은 전체 텍스트
            
            if current_context_law:
                # 현재 기준 법령이 언급된 청크(Chunk)만 필터링
                filtered_chunks = [
                    chunk for chunk in rag_chunks
                    if current_context_law in chunk.get("text", "") or 
                       any(current_context_law in str(v) for v in chunk.get("metadata", {}).values())
                ]
                
                # 필터링된 청크가 있다면, 그 구역 안에서만 검증 수행
                if filtered_chunks:
                    target_rag_text = " ".join([chunk.get("text", "") for chunk in filtered_chunks])
                else:
                    # 💡 [게이트 2] 앵커 불일치면 전체 RAG fallback을 금지하여 오교정 방지
                    if word not in full_rag_text:
                        hallucinations.append({
                            "label": label,
                            "wrong_word": word,
                            "correct_word": None,
                            "context_law": current_context_law,
                            "start": start,
                            "end": end,
                            "reason_code": "ANCHOR_MISMATCH_BLOCKED",
                            "reason": "답변 기준 법령 앵커와 일치하는 RAG 청크가 없어 자동 교정을 차단했습니다.",
                        })
                    continue

            # 💡 [핵심 로직 3] 타겟 구역 내 팩트체크 및 교정 단어 탐색
            if word not in target_rag_text:
                correct_candidate = None
                
                # 틀렸다면, 해당 타겟 구역에서 '같은 라벨'을 가진 정답 후보를 찾음
                rag_entities = self.extract_entities(target_rag_text)
                correct_candidate = self._choose_candidate(label, word, rag_entities)
                
                hallucinations.append({
                    "label": label,
                    "wrong_word": word,
                    "correct_word": correct_candidate,
                    "context_law": current_context_law, # 어떤 법령 문맥에서 틀렸는지 기록
                    "start": start,
                    "end": end,
                    "reason_code": "FACT_MISMATCH" if correct_candidate else "FACT_MISMATCH_NO_CANDIDATE",
                    "reason": "컨텍스트 앵커 기준 RAG 범위에서 동일 사실을 찾지 못해 교정 후보를 탐색했습니다.",
                })
                
        return hallucinations

    def _detect_domains(self, text):
        detected = set()
        normalized = re.sub(r"\s+", "", text or "")
        for domain, keywords in self.domain_keywords.items():
            for kw in keywords:
                if kw in normalized:
                    detected.add(domain)
                    break
        return detected

    def _is_domain_aligned(self, answer_domains, rag_domains):
        # 키워드가 한쪽이라도 비어 있으면 보수적으로 허용하고, 둘 다 있을 때만 교집합을 요구
        if not answer_domains or not rag_domains:
            return True
        return bool(answer_domains & rag_domains)

    def _is_reliable_law_anchor(self, text):
        if not text:
            return False
        token = text.strip()
        if len(token) < 2:
            return False
        if token.isdigit():
            return False

        has_korean = any("가" <= ch <= "힣" for ch in token)
        if not has_korean:
            return False

        legal_hint = ["법", "령", "규칙", "조", "헌법", "민법", "형법"]
        return any(h in token for h in legal_hint)

    def _extract_law_anchors_from_text(self, text):
        if not text:
            return []

        pattern = re.compile(r"([가-힣A-Za-z0-9\s]{2,40}(?:법률|시행규칙|시행령|규칙|법))")
        anchors = []
        for m in pattern.finditer(text):
            anchor = re.sub(r"\s+", " ", m.group(1)).strip()
            if self._is_reliable_law_anchor(anchor):
                anchors.append(anchor)

        dedup = []
        seen = set()
        for a in anchors:
            if a in seen:
                continue
            seen.add(a)
            dedup.append(a)
        return dedup

    def _choose_candidate(self, label, wrong_word, rag_entities):
        same_label_words = [
            ent.get("word", "")
            for ent in rag_entities
            if ent.get("entity_group") == label and isinstance(ent.get("word"), str)
        ]
        same_label_words = [w.strip() for w in same_label_words if w and w.strip()]
        same_label_words = [w for w in same_label_words if len(w) >= 2]

        if not same_label_words:
            return None

        if label == "DATE":
            date_words = [w for w in same_label_words if any(ch.isdigit() for ch in w)]
            if not date_words:
                return None

            # 날짜는 단위/형식을 우선 맞춘 후보를 사용해 과교정 위험을 줄인다.
            if "일" in wrong_word:
                for w in date_words:
                    if "일" in w:
                        return w

            if wrong_word.isdigit():
                for w in date_words:
                    if "일" in w:
                        return w
                return None

            # 숫자/단위가 없는 모호한 DATE 토큰(예: "이내")은 자동 교정하지 않는다.
            if not any(ch.isdigit() for ch in wrong_word):
                return None

            return date_words[0]

        return same_label_words[0]

    async def check_and_correct(self, answer, rag_docs):
        hallucinations = self.find_hallucinations(answer, rag_docs)
        corrected_answer = self._corrector.fix_answer(answer, hallucinations)
        return NERCheckResult(
            original_answer=answer,
            corrected_answer=corrected_answer,
            found_entities=self.extract_entities(answer),
            mismatched_entities=hallucinations,
            was_corrected=corrected_answer != answer,
        )


# Team_code pipeline 호환용 전역 인스턴스
ner_checker = NERFactChecker()