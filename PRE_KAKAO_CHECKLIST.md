# Pre-Kakao Integration Checklist (LawsGuard)

## 1) Pin model version
Set model version explicitly before running the pipeline.

PowerShell:

$env:LAWSGUARD_NER_MODEL='outputs/legal-ner-lawsguard-v1-gpu'

## 2) Run operational scenario test (real RAG + actual LLM answer)
Use real RAG chunks and pass the actual LLM answer text.

c:/GitHub/chatbot/.venv/Scripts/python.exe run_operational_scenario.py --rag-file data/real_data/New_Dataset/rag_law_chunks.jsonl --answer "고용보험법 시행령 제1조에 따르면 신청은 60일 이내에 해야 합니다."

Optional:
- Use --chunk-id to target a specific chunk
- Use --max-context-chunks to control retrieval context size

## 3) Generate label-wise entity evaluation report
Sample (fast check):

c:/GitHub/chatbot/.venv/Scripts/python.exe evaluate_entity_report.py --data-dirs data/hallucination_data data/real_bio_data --max-samples 2000 --output-json outputs/legal-ner-lawsguard-v1-gpu/entity_report_sample2000.json

Full dataset:

c:/GitHub/chatbot/.venv/Scripts/python.exe evaluate_entity_report.py --data-dirs data/hallucination_data data/real_bio_data --output-json outputs/legal-ner-lawsguard-v1-gpu/entity_report_full.json

## 4) Interpret pipeline status
- PASS: no hallucination detected
- DETECTED: hallucination detected but no safe replacement candidate
- CORRECTED: at least one entity safely replaced

## 5) Ready for Kakao integration
When the above checks are green, return result.final_answer in Kakao response JSON.
