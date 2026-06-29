"""classifier_harness — data_classifier.py 자동 평가·최적화 하네스.

목적
----
문서 등급 분류 모델(data_classifier.py, 정규식 + Presidio NER + BERT 신경망 3-tier)이
외부 LLM/GPU 없이 LLM 수준 성능에 도달하도록, 등급(C/S/O)·포맷별 합성 테스트셋을 생성하고,
모델을 분류·KPI 측정한 뒤, 정규식·NER·키워드·뉴럴 백엔드·앙상블·임계값을 바꿔가며
최적 설정을 자동 탐색한다. 비교군으로 Claude(LLM) 분류를 수행해 BERT-모델과 정량 비교한다.

파이프라인
----------
    generate → detect(캐시) → score-sweep → metrics → optimize → llm-baseline → report

비싼 단계(추출+NER+뉴럴)는 1회 캐시하고, 싼 단계(점수·앙상블)는 캐시 위에서 수천 조합을
스윕한다. 결과는 SQLite(results.db) + Excel 로 정리한다.
"""
__all__ = ["synth", "corpus", "db", "detect", "score", "metrics", "optimize", "report", "llm_baseline"]
