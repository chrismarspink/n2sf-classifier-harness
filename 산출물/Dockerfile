# N²SF 등급분류 서비스 — 온디바이스(CPU) 분류 API 컨테이너
# 빌드: docker build -t innotium/n2sf-classifier:1.0 -f 산출물/Dockerfile .
# 실행: docker run -p 8080:8080 innotium/n2sf-classifier:1.0
#   분류: curl -s localhost:8080/classify -d '{"text":"주민번호 900101-1234567"}'
FROM python:3.12-slim

WORKDIR /app

# 시스템 의존(빌드 도구 최소)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 파이썬 의존(CPU 전용 torch)
COPY 산출물/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -m spacy download ko_core_news_lg

# 엔진 + 참조 소스 + 모델
COPY data_classifier.py /app/data_classifier.py
COPY 산출물/classify_n2sf.py /app/classify_n2sf.py
COPY 산출물/serve.py /app/serve.py
# 모델: 기본 base(균형). 정확도 최우선이면 n2sf-xlmr-large도 복사하고 N2SF_MODEL 지정.
COPY 산출물/models/n2sf-base   /app/models/n2sf-base
COPY 산출물/models/n2sf-small  /app/models/n2sf-small
# COPY 산출물/models/n2sf-xlmr-large /app/models/n2sf-xlmr-large

ENV N2SF_MODEL=n2sf-base
ENV PORT=8080
EXPOSE 8080

# 추론 100% 로컬 CPU — 외부 LLM/네트워크 호출 없음
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"
CMD ["python", "serve.py"]
