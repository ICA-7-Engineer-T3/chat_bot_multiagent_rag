# 뭉이 멀티에이전트 챗봇 (chat_bot_multiagent_rag)

Flask 기반 **뭉이(Moong)** 멀티에이전트 챗봇입니다.  
Router → (Analyzer / Memory) → Selector → Writer → Guardrail 파이프라인으로 동작하며, RAG 기반 감정 분석과 페르소나별 답변 생성, GUI에서 에이전트별 결과·실행 시간을 확인할 수 있습니다.

---

## 1. 알고리즘 개요

### 1.1 전체 구조

한 번의 사용자 메시지는 **Router**에서 분류된 뒤, 필요에 따라 Analyzer·Memory를 거쳐 Selector → Writer → Guardrail 순으로 처리됩니다.

```
[사용자 입력] → Router → ┬─ (단순) → Selector → Writer → Guardrail → [답변 or 재작성]
                         └─ (복잡) → Analyzer and/or Memory → Selector → Writer → Guardrail
                                                                    ↑___________|
                                                                    (REJECT 시 Writer 재진입)
```

- **LangGraph**로 StateGraph를 정의하고, 비동기 노드(`ainvoke`)로 워크플로 실행.
- **LLM**: OpenRouter API (`google/gemini-2.0-flash-lite-001`) 사용. API 키는 웹에서 입력하거나 `GEMINI_API_KEY` 환경 변수로 지정.
- **RAG**: SentenceTransformer + FAISS로 감정 대화 DB에서 유사 사례 검색 후 Analyzer에 전달.

### 1.2 에이전트 역할

| 단계 | 에이전트 | 역할 |
|------|----------|------|
| **Router** | Moong-Router | 입력이 단순(인사/리액션)인지, 감정·의도 분석(`needs_analyzer`)·과거 맥락(`needs_memory`)이 필요한지 JSON으로 분류. |
| **Analyzer** | Moong-Scanner | RAG 유사 대화 데이터 + 사용자 입력으로 감정·의도 분석, 페르소나 추천(JSON). |
| **Memory** | Moong-Memory | 최근 대화 기록으로 맥락 연결, 감정 추이, 필수 사실 정리(JSON). |
| **Selector** | — | 사용자가 선택한 페르소나(mate / guide / pet)와 해당 페르소나 프롬프트를 상태에 설정. |
| **Writer** | Moong | Analyzer·Memory 결과와 페르소나 가이드라인으로 답변 초안 작성. |
| **Guardrail** | Moong-Guardrail | 분량·페르소나 유지·비판/진단 여부 검수. `APPROVE` / `REJECT`(사유) → REJECT 시 Writer로 재진입. |

### 1.3 RAG(감정 유사도 검색)

- **입력**: 사용자 발화 한 문장.
- **처리**: SentenceTransformer(`jhgan/ko-sroberta-multitask`)로 임베딩 → FAISS 인덱스에서 Top-K(30) 유사 대화 검색 → 감정 요약·주요 감정·유사 상황 3건을 Analyzer에 전달.
- **데이터**: SQLite `dialogues` 테이블(컬럼: `emotion_middle_class`, `emotion_low_class`, `emotion_high_class`, `sing_turn_text`)과 미리 구축한 FAISS 인덱스. 경로는 `DBs/` 폴더 기준 상대 경로 사용.

---

## 2. 환경 설정

### 2.1 필요 사항

- **Python**: 3.10 이상 권장
- **API 키**: OpenRouter API 키 ([OpenRouter](https://openrouter.ai/)에서 발급). 웹 UI에서 입력하거나 `GEMINI_API_KEY` 환경 변수에 설정.
- **RAG 데이터**: `DBs/` 폴더에 다음 파일 배치 시 상대 경로로 자동 참조.
  - `DBs/dialogues_lite.db` — SQLite DB (`dialogues` 테이블)
  - `DBs/faiss_index_pq.faiss` — FAISS 인덱스

### 2.2 패키지 설치

```bash
cd chat_bot_multiagent_rag
pip install -r requirements.txt
```

`requirements.txt`에 포함된 패키지:

- **Flask** — 웹 서버, 라우트, 세션
- **faiss-cpu** — RAG 유사도 검색
- **numpy**, **pandas** — 벡터·DB 조회
- **langchain-google-genai**, **langchain-openai** — LLM (OpenRouter는 langchain-openai 사용)
- **langchain-core** — 메시지 타입 등
- **langgraph** — StateGraph, 워크플로
- **sentence-transformers** — 한국어 임베딩 모델

### 2.3 환경 변수 (선택)

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `GEMINI_API_KEY` | OpenRouter(또는 사용 중인 API) 키. 미설정 시 웹에서 입력 | — |
| `FLASK_SECRET_KEY` | Flask 세션 암호화 | `moong-multiagent-secret-change-in-production` |
| `MOONG_DB_PATH` | 대화 DB SQLite 경로 | `DBs/dialogues_lite.db` (상대 경로) |
| `MOONG_FAISS_INDEX_PATH` | FAISS 인덱스 경로 | `DBs/faiss_index_pq.faiss` (상대 경로) |

---

## 3. 실행 방법

### 3.1 서버 실행

```bash
python main.py
```

- 콘솔에 `Flask 서버 시작 (챗봇 + 에이전트 분석 GUI) http://0.0.0.0:5000` 출력.
- 포트 **5000**에서 동작.

### 3.2 브라우저 접속

- **http://localhost:5000** 접속.
- API 키 미설정 시 첫 화면에서 API 키 입력 후 챗봇 사용.
- 페르소나(mate / guide / pet) 선택 후 메시지 전송 → 채팅 + 오른쪽 패널에서 에이전트별 분석 결과·실행 시간 확인.

### 3.3 API 키 없이 실행 시

- 서버는 기동되며, 웹에서 API 키를 입력하면 그 시점에 LLM·워크플로가 초기화됩니다.
- DB·FAISS·SentenceTransformer는 API 키 설정 또는 첫 채팅 시 1회 로드됩니다.

---

## 4. API 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 채팅 GUI (moong_multiagent_chat.html) |
| GET | `/config` | API 키 설정 여부 (`api_key_configured`) |
| POST | `/api-key` | API 키 등록 및 모델 초기화 |
| POST | `/chat` | 메시지·페르소나 전달 → 답변 + 에이전트 결과 JSON |
| POST | `/reset` | 대화·에이전트 결과 세션 초기화 |
| GET | `/history` | 현재 대화·마지막 에이전트 결과 |

---

## 5. 요약

| 항목 | 내용 |
|------|------|
| **알고리즘** | Router → (Analyzer / Memory) → Selector → Writer → Guardrail, RAG(FAISS + ko-sroberta-multitask) |
| **LLM** | OpenRouter `google/gemini-2.0-flash-lite-001`, 페르소나별 temperature(mate 0.7, guide 0.4, pet 0.8) |
| **RAG 데이터** | `DBs/dialogues_lite.db`, `DBs/faiss_index_pq.faiss` (상대 경로) |
| **환경** | Python 3.10+, `pip install -r requirements.txt` |
| **실행** | `python main.py` → 브라우저에서 `http://localhost:5000` |
