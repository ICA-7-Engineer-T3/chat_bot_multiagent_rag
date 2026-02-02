# 뭉이 멀티에이전트 챗봇 — 가이드

Flask 기반 **뭉이(Moong)** 멀티에이전트 챗봇의 알고리즘 개요, 환경 설정, 실행 방법을 정리한 문서입니다.

---

## 1. 알고리즘 개요

### 1.1 전체 구조

챗봇은 **5단계 에이전트 파이프라인**으로 동작합니다.  
한 번의 사용자 메시지가 들어오면 아래 순서로 처리된 뒤 최종 답변이 생성됩니다.

```
[사용자 입력] → Analyzer → Memory → Selector → Writer → Guardrail → [답변 or 재작성]
                                                    ↑_______________|
                                                    (REJECT 시 Writer로 재진입)
```

- **LangGraph**로 상태 그래프(StateGraph)를 정의하고, 각 단계를 노드로 연결합니다.
- **Gemini**(Google Generative AI)를 LLM으로 사용합니다.
- **RAG(Retrieval-Augmented Generation)**로 감정 대화 DB에서 유사 사례를 검색해 분석에 활용합니다.

### 1.2 각 에이전트 역할

| 단계 | 에이전트명 | 역할 |
|------|------------|------|
| **Analyzer** | Moong-Scanner | 사용자 입력 + RAG 유사 대화 데이터로 **감정·의도 분석**. 감정 벡터, 숨은 니즈, 페르소나 추천을 JSON 형태로 출력. |
| **Memory** | Moong-Memory | **대화 기록**을 보고 맥락 연결, 감정 추이, 필수 사실을 정리. 새 이슈 vs 이전 대화 연장 여부 판단. |
| **Selector** | — | 사용자가 선택한 **페르소나**(mate / guide / pet)를 상태에 유지. |
| **Writer** | Moong (뭉) | Analyzer·Memory 결과와 페르소나에 맞춰 **답변 초안** 작성. (mate=친구형, guide=멘탈코치형, pet=강아지형) |
| **Guardrail** | Moong-Guardrail | 답변을 **검수**(3줄 이내, 페르소나 유지, 비판/진단 없음). 적절하면 `APPROVE`, 부적절하면 `REJECT` + 사유 반환 → Writer로 재진입. |

### 1.3 RAG(감정 유사도 검색)

- **입력**: 사용자 발화 한 문장.
- **처리**:
  1. **SentenceTransformer** (`jhgan/ko-sroberta-multitask`)로 문장 임베딩 생성.
  2. **FAISS** 인덱스에서 Top-K(기본 30) 유사 대화 검색.
  3. 검색된 행의 `emotion_middle_class`, `emotion_low_class`, `emotion_high_class`, `sing_turn_text`를 집계.
  4. 감정 비율(15% 이상만 요약), 주요 감정, 유사 상황 문장 3개를 정리해 Analyzer에 전달.
- **데이터**: 프로젝트의 `DBs/` 폴더에 있는 SQLite DB(`dialogues` 테이블) + 임베딩(`.npy`) + FAISS 인덱스(`.faiss`)를 상대 경로로 참조합니다.

### 1.4 상태(State)와 워크플로

- **MoongState**: `messages`, `analyzer_output`, `analyzer_rag_result`, `memory_context`, `selected_persona`, `draft_answer`, `guardrail_status`, `review_feedback` 등.
- Guardrail에서 **REJECT**이면 Writer로만 돌아가고, **APPROVE**이면 워크플로 종료 후 해당 `draft_answer`가 사용자에게 전달됩니다.

---

## 2. 환경 설정

### 2.1 필요 사항

- **Python**: 3.10 이상 권장
- **Gemini API 키**: [Google AI Studio](https://aistudio.google.com/)에서 발급
- **RAG용 데이터**: 프로젝트 루트의 **`DBs`** 폴더에 아래 세 파일을 두면 **상대 경로**로 자동 참조됩니다.
  - `DBs/total_data.db` — SQLite DB (`dialogues` 테이블, 컬럼: `emotion_middle_class`, `emotion_low_class`, `emotion_high_class`, `sing_turn_text`)
  - `DBs/faiss_embeddings.npy` — 임베딩 배열
  - `DBs/faiss_index.faiss` — FAISS 인덱스

### 2.2 프로젝트 클론/이동

```bash
cd c:\chat_bot_multiagent_rag
```

### 2.3 가상환경 생성 및 활성화 (권장)

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.4 패키지 설치

```bash
pip install -r requirements.txt
```

### 2.5 환경 변수 (선택)

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `GEMINI_API_KEY` | Gemini API 키 (설정 시 앱 시작 시 자동 초기화) | (없으면 웹에서 입력) |
| `FLASK_SECRET_KEY` | Flask 세션 암호화용 | `moong-multiagent-secret-change-in-production` |
| `MOONG_DB_PATH` | 대화 DB SQLite 파일 경로 | `DBs/total_data.db` (프로젝트 기준 상대 경로) |
| `MOONG_EMBEDDINGS_PATH` | 임베딩 `.npy` 파일 경로 | `DBs/faiss_embeddings.npy` |
| `MOONG_FAISS_INDEX_PATH` | FAISS 인덱스 `.faiss` 파일 경로 | `DBs/faiss_index.faiss` |

- **기본 동작**: 위 세 경로는 **설정하지 않으면** `main.py`가 있는 디렉터리 기준 **`DBs/`** 폴더를 참조합니다. `DBs` 안에 `total_data.db`, `faiss_embeddings.npy`, `faiss_index.faiss`를 두면 별도 설정 없이 동작합니다.
- **다른 경로를 쓰고 싶을 때만** 아래처럼 환경 변수로 덮어쓰면 됩니다.

**Windows (PowerShell, 현재 세션):**

```powershell
$env:GEMINI_API_KEY = "여기에_API_키"
# DB/임베딩/인덱스를 다른 위치에 둔 경우에만:
# $env:MOONG_DB_PATH = "C:\data\total_data.db"
# $env:MOONG_EMBEDDINGS_PATH = "C:\data\faiss_embeddings.npy"
# $env:MOONG_FAISS_INDEX_PATH = "C:\data\faiss_index.faiss"
```

**Linux / macOS:**

```bash
export GEMINI_API_KEY="여기에_API_키"
# DB/임베딩/인덱스를 다른 위치에 둔 경우에만:
# export MOONG_DB_PATH="/path/to/total_data.db"
# export MOONG_EMBEDDINGS_PATH="/path/to/faiss_embeddings.npy"
# export MOONG_FAISS_INDEX_PATH="/path/to/faiss_index.faiss"
```

- `GEMINI_API_KEY`를 넣지 않으면, 실행 후 웹 UI에서 API 키를 입력해야 챗봇이 동작합니다.

---

## 3. 실행 방법

### 3.1 서버 실행

```bash
python main.py
```

- 콘솔에 `Flask 서버 시작 (챗봇 + 에이전트 분석 GUI) http://0.0.0.0:5000` 및 포트 안내가 나옵니다.
- 실제 리스닝 포트는 코드 기준 **5000** (`app.run(..., port=5000)`).

### 3.2 브라우저에서 접속

- 주소: **http://localhost:5000** (같은 PC) 또는 **http://0.0.0.0:5000**
- 첫 화면에서 API 키가 설정되지 않았으면 **Gemini API 키**를 입력한 뒤, 챗봇과 대화할 수 있습니다.
- 페르소나(mate / guide / pet) 선택 후 메시지를 보내면, Analyzer·Memory·Writer·Guardrail 등 에이전트 결과를 GUI에서 확인할 수 있습니다.

### 3.3 API 키 없이 실행했을 때

- 서버는 정상 기동되며, 웹에서 API 키를 입력하면 그 시점에 LLM·워크플로가 초기화됩니다.
- DB/FAISS/임베딩은 **첫 채팅 시** 또는 API 키 설정 시 한 번만 로드됩니다.

---

## 4. 요약

| 항목 | 내용 |
|------|------|
| **알고리즘** | Analyzer → Memory → Selector → Writer → Guardrail 5단계 멀티에이전트 + RAG(FAISS + 한국어 SBERT) |
| **RAG 데이터** | `DBs/total_data.db`, `DBs/faiss_embeddings.npy`, `DBs/faiss_index.faiss` (상대 경로, 환경 변수로 덮어쓰기 가능) |
| **환경** | Python 3.10+, `pip install -r requirements.txt`, (선택) 환경 변수로 API 키·경로 설정 |
| **실행** | `python main.py` 후 브라우저에서 `http://localhost:5000` 접속 |
