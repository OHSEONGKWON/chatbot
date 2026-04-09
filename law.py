import requests
import json
import time
import os

# ==========================================
# 설정(Configuration)
# ==========================================
API_KEY = 'tkatjdrnt123'  # 본인의 인증키(OC)로 변경하세요 (test는 임시 키)
SEARCH_QUERY = '근로기준법'  # 수집할 검색어
MAX_PAGES = 3      # 수집할 최대 페이지 수 (페이지당 20개)
OUTPUT_FILE = 'supreme_court_precedents.jsonl' # 저장될 파일명

# ==========================================
# 1. 판례 목록(일련번호) 수집 함수
# ==========================================
def get_precedent_ids(query, page=1):
    url = "http://www.law.go.kr/DRF/lawSearch.do"
    params = {
        'OC': API_KEY,
        'target': 'prec',
        'type': 'JSON',
        'query': query,
        'page': page
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 검색 결과가 없는 경우
        if 'PrecSearch' not in data or 'prec' not in data['PrecSearch']:
            return []
            
        prec_list = data['PrecSearch']['prec']
        # 판례 일련번호(판례일련번호)만 추출
        return [item['판례일련번호'] for item in prec_list]
        
    except Exception as e:
        print(f"[목록 조회 에러] 페이지 {page}: {e}")
        return []

# ==========================================
# 2. 판례 상세 내용(원문) 수집 함수
# ==========================================
def get_precedent_detail(prec_id):
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {
        'OC': API_KEY,
        'target': 'prec',
        'type': 'JSON',
        'ID': prec_id
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'PrecService' not in data:
            return None
            
        prec_detail = data['PrecService']
        
        # LLM 데이터셋이나 검색 서버 구축에 필요한 핵심 데이터만 파싱하여 딕셔너리로 구성
        dataset_item = {
            "판례일련번호": prec_detail.get('판례정보일련번호', ''),
            "사건명": prec_detail.get('사건명', ''),
            "사건번호": prec_detail.get('사건번호', ''),
            "선고일자": prec_detail.get('선고일자', ''),
            "법원명": prec_detail.get('법원명', ''),
            "사건종류명": prec_detail.get('사건종류명', ''),
            "판결유형": prec_detail.get('판결유형', ''),
            "판시사항": prec_detail.get('판시사항', ''),
            "판결요지": prec_detail.get('판결요지', ''),
            "참조조문": prec_detail.get('참조조문', ''),
            "참조판례": prec_detail.get('참조판례', ''),
            "판례내용": prec_detail.get('판례내용', '') # 전체 판결문 원문
        }
        return dataset_item
        
    except Exception as e:
        print(f"[상세 조회 에러] ID {prec_id}: {e}")
        return None

# ==========================================
# 3. 메인 실행 로직 (데이터셋 구축)
# ==========================================
def main():
    print(f"[{SEARCH_QUERY}] 관련 판례 수집을 시작합니다...")
    total_collected = 0
    
    # 파일을 덮어쓰지 않고 이어서 쓰기(append) 모드로 오픈
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for page in range(1, MAX_PAGES + 1):
            print(f"--- {page}페이지 목록 수집 중 ---")
            prec_ids = get_precedent_ids(SEARCH_QUERY, page)
            
            if not prec_ids:
                print("더 이상 수집할 판례가 없습니다.")
                break
                
            for prec_id in prec_ids:
                # API 서버에 과부하를 주지 않기 위한 딜레이 (매우 중요)
                time.sleep(0.2) 
                
                detail = get_precedent_detail(prec_id)
                if detail:
                    # 대법원 판례만 필터링하여 저장
                    if detail.get("법원명") == "대법원":
                        # 딕셔너리를 JSON 문자열로 변환하여 파일에 한 줄씩 기록 (JSONL 포맷)
                        json_str = json.dumps(detail, ensure_ascii=False)
                        f.write(json_str + '\n')
                        total_collected += 1
                        print(f"수집 완료: {detail['사건명']} ({detail['사건번호']})")
                    else:
                        print(f"스킵됨 (대법원 아님): 법원명 - {detail.get('법원명')}")

    print("==========================================")
    print(f"수집 완료! 총 {total_collected}개의 대법원 판례가 {OUTPUT_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    main()