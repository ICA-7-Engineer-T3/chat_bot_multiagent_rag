# Flask 기반 뭉이 멀티에이전트 챗봇
# moong_multiagent.ipynb 파이프라인(Scanner → Memory → Selector → Writer → Guardrail)을
# 챗봇 형태로 실행하고, 에이전트별 분석 결과를 GUI에서 확인할 수 있음.

import os
import time
import sqlite3
import pandas as pd
import numpy as np
import faiss
from collections import Counter
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "moong-multiagent-secret-change-in-production")

# ---------------------------------------------------------------------------
# 환경 변수 (GEMINI_API_KEY는 env 또는 웹에서 입력)
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DBS_DIR = os.path.join(_BASE_DIR, "DBs")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_PATH = os.environ.get("MOONG_DB_PATH", os.path.join(_DBS_DIR, "dialogues_lite.db"))
FAISS_INDEX_PATH = os.environ.get("MOONG_FAISS_INDEX_PATH", os.path.join(_DBS_DIR, "faiss_index_pq.faiss"))

# ---------------------------------------------------------------------------
# 전역 리소스 (앱 시작 시 로드)
# ---------------------------------------------------------------------------
llm = None
_current_api_key: Optional[str] = None  # 웹/환경에서 설정한 API 키 (페르소나별 LLM 생성용)
sbert_model = None
index = None
df = None
workflow_app = None


# ---------------------------------------------------------------------------
# State 정의 (노트북 + analyzer_rag_result for GUI)
# ---------------------------------------------------------------------------
class MoongState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    analyzer_output: str
    analyzer_rag_result: dict
    memory_context: str
    selected_persona: str
    draft_answer: str
    guardrail_status: Literal["APPROVE", "REJECT"]
    review_feedback: str
    # 각 에이전트별 실행 시간 (ms)
    analyzer_time_ms: float
    memory_time_ms: float
    selector_time_ms: float
    writer_time_ms: float
    guardrail_time_ms: float


# ---------------------------------------------------------------------------
# RAG 도구 (노트북과 동일)
# ---------------------------------------------------------------------------
def multi_emotion_analysis_agent_func(
    user_input: str,
    sbert_model,
    index,
    df: pd.DataFrame,
) -> dict:
    """입력과 유사한 대화들의 감정 분포 분석, 복합 감정 리포트 및 유사 상황 리턴."""
    # MODIFY: 변경 필요 (TOP_K 값 변경)
    TOP_K = 15
    query_vec = sbert_model.encode([user_input]).astype("float32")
    _, indices = index.search(query_vec, TOP_K)

    found_middle_emotions = []
    found_low_emotions = []
    found_high_emotions = []
    found_single_turn_texts = []
    for i in indices[0]:
        if 0 <= i < len(df):
            row = df.iloc[i]
            middle = row["emotion_middle_class"]
            low = row["emotion_low_class"]
            high = row["emotion_high_class"]
            single_turn = row["sing_turn_text"]
            found_middle_emotions.append(
                middle if pd.notnull(middle) and str(middle).strip() != "" else "_없음"
            )
            found_low_emotions.append(
                low if pd.notnull(low) and str(low).strip() != "" else "_없음"
            )
            found_high_emotions.append(
                high if pd.notnull(high) and str(high).strip() != "" else "_없음"
            )
            found_single_turn_texts.append(
                single_turn if pd.notnull(single_turn) else ""
            )

    counters = {
        "middle": Counter(found_middle_emotions),
        "low": Counter(found_low_emotions),
        "high": Counter(found_high_emotions),
    }
    totals = {k: sum(v.values()) for k, v in counters.items()}

    emotion_summaries = {}
    primary_emotions = {}
    for level in ["middle", "low", "high"]:
        report_list = []
        prim_emos = []
        total = totals[level]
        counter = counters[level]
        emo_counts = [
            (emo, cnt) for emo, cnt in counter.most_common() if emo != "_없음"
        ]
        for emo, cnt in emo_counts:
            percent = (cnt / total) * 100 if total > 0 else 0
            if percent >= 15:
                report_list.append(f"{emo}({percent:.0f}%)")
                prim_emos.append(emo)
        emotion_summaries[level] = ", ".join(report_list)
        primary_emotions[level] = prim_emos

    return {
        "emotion_summary": emotion_summaries,
        "primary_emotions": primary_emotions,
        "similar_situations": [
            s for s in found_single_turn_texts if s and str(s).strip() != ""
        ][:3],
    }


def _llm_content(response) -> str:
    """LLM invoke 응답에서 텍스트 추출"""
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------
def analyzer_node(state: MoongState, config=None) -> dict:
    start = time.perf_counter()
    user_input = state["messages"][-1].content
    emotion_rag_result = multi_emotion_analysis_agent_func(
        user_input=user_input,
        sbert_model=sbert_model,
        index=index,
        df=df,
    )
    scanner_prompt = f"""
        # Role
        너는 사용자의 감정과 의도를 정밀하게 분석하는 전문 심리 분석 에이전트 'Moong-Scanner'이다.
        단순한 텍스트 분석을 넘어, 제공된 유사 사례와 통계 데이터를 바탕으로 사용자의 숨은 의도를 파악하라.

        # Input
        - 사용자 입력: {user_input}
        - RAG 유사도 높은 데이터: {emotion_rag_result}

        # Task: Step-by-Step Analysis
        1. 감정 벡터 추출: RAG 데이터의 감정 퍼센테이지를 가중치로 활용하여 현재 사용자의 감정 상태를 상위 3개까지 도출하라.
        2. 의도 심층 분석: 과거 유사 사례의 '실제 의도'와 현재 입력을 비교하여 사용자가 직접적으로 말하지 않은 '숨은 니즈'를 식별하라.
        3. 페르소나 가이드 생성: 다음 단계인 Persona-Selector가 최적의 모드를 선택할 수 있도록 분석 요약본을 전달하라.

        # Output Format (JSON)
        {{
            "primary_emotion": {{ "label": "string", "confidence": "float" }},
            "detected_intent": "string",
            "context_match_score": "float (0.0~1.0)",
            "persona_recommendation": "string (e.g., 위로형, 조언형, 일상형)"
        }}

        # Constraint
        - RAG 데이터와 현재 입력이 충돌할 경우, 현재 입력의 최신 맥락을 우선하되 RAG의 통계적 경향성을 참고치로 명시할 것.
        - 감정 분석 시 중립적인 태도를 유지하며 과도한 추측은 지양할 것.
    """
    analyzer_start = time.perf_counter()
    analyzer_output = _llm_content(llm.invoke(scanner_prompt))
    end = time.perf_counter()
    return {
        "analyzer_output": analyzer_output,
        "analyzer_rag_result": emotion_rag_result,
        "analyzer_time_ms": (end - start) * 1000.0,
    }


def memory_node(state: MoongState, config=None) -> dict:
    start = time.perf_counter()
    last_msg = state["messages"][-1]
    msgs = state["messages"]
    memory_prompt = f"""
        # Role
        너는 사용자의 과거와 현재를 잇는 기억 관리자 'Moong-Memory'이다.
        단순한 기록 조회를 넘어, 대화의 '흐름'과 '사용자의 변화'를 포착하여 현재 대화에 생명력을 불어넣는다.

        # Input
        - 사용자 입력: {last_msg}
        - 대화 기록: {msgs}

        # Task: Contextual Analysis
        1. **맥락 연결 (Contextual Link)**: 현재 사용자가 하는 말이 과거에 언급했던 특정 사건, 인물, 감정의 연장선상에 있는지 판단하라.
        2. **상태 변화 추적 (State Tracking)**: 과거 대비 사용자의 감정 수치나 태도가 어떻게 변했는지 분석하라 (예: 개선됨, 악화됨, 유지됨).
        3. **핵심 키워드 추출**: 답변에 반드시 포함해야 할 과거의 고유 명사나 에피소드를 선별하라.

        # Output Format (JSON)
        {{
            "is_returning_issue": "boolean",
            "memory_summary": "string",
            "emotional_trend": "string",
            "essential_facts": ["string"]
        }}

        # Constraint
        - 과거 기억이 현재 대화와 전혀 관련 없다면 "새로운 대화 맥락"으로 정의하고 억지로 엮지 말 것.
        - 사용자가 잊고 싶어 하는 부정적인 기억을 무분별하게 상기시키지 않도록 주의할 것.
    """
    memory_output = _llm_content(llm.invoke(memory_prompt))
    end = time.perf_counter()
    return {
        "memory_context": memory_output,
        "memory_time_ms": (end - start) * 1000.0,
    }


def selector_node(state: MoongState, config=None) -> dict:
    start = time.perf_counter()
    current_persona = state.get("selected_persona", "mate")
    end = time.perf_counter()
    return {
        "selected_persona": current_persona,
        "selector_time_ms": (end - start) * 1000.0,
    }


def persona_writer_node(state: MoongState, config=None) -> dict:
    # MODIFY: 변경 필요 (페르소나 프롬프트 내용 변경)
    feedback = state.get("review_feedback", "")
    analyzer_out = state.get("analyzer_output", "")
    memory_ctx = state.get("memory_context", "")

    if state.get("selected_persona") == "mate":
        persona_prompt = f"""
            # Role
            당신은 사용자의 단짝 친구 '메이트 뭉'입니다.
            # Guidelines
            1. 호칭: 야, 너라고 편하게 부르세요.
            2. 말투: 유행어(갓생, 국룰 등)를 섞은 짧은 반말을 사용하세요. 답변 끝에는 반드시 질문을 포함하세요.
            3. 미션: 자기 경험을 덧붙여 티키타카를 만드세요. (예: "그건 국룰이지 ㅋㅋ 넌 어떻게 생각해?")
            4. 제약: 너무 진지해지지 마세요. 즐거운 에너지를 유지합니다.
            # Reference
            - 감정/의도 분석 결과: {analyzer_out}
            - 과거 맥락: {memory_ctx}
            - 가드레일 피드백: {feedback}
        """
    elif state.get("selected_persona") == "guide":
        persona_prompt = f"""
            # Role
            당신은 사용자의 일상을 가이드하는 '가이드 뭉'입니다.
            # Guidelines
            1. 호칭: 고객님이라고 정중히 부르세요.
            2. 말투: 정중한 경어체를 사용하며, 미사여구 없이 담백하게 핵심만 말하세요.
            3. 미션: 심리학 용어 없이 상황을 요약하고 사소한 실천(환기, 메모 등)을 제안하세요.
            4. 제약: 전문적인 상담사처럼 굴지 마세요. 든든한 조력자 수준을 유지합니다.
            # Reference
            - 감정/의도 분석 결과: {analyzer_out}
            - 과거 맥락: {memory_ctx}
            - 가드레일 피드백: {feedback}
        """
    elif state.get("selected_persona") == "pet":
        persona_prompt = f"""
            # Role
            당신은 사용자의 반려동물 '펫 뭉'입니다.
            # Guidelines
            1. 호칭: 항상 `주인님`이라고 부르세요.
            2. 말투: 짧은 반말과 의성어/의태어(뭉뭉)를 사용하세요. 답변은 2문장 이내로 제한합니다.
            3. 금기: 절대 충고나 조언을 하지 마세요. 사용자가 화를 내도 애교로 대응합니다.
            4. 미션: 사용자의 감정을 그대로 따라 하세요. (예: "슬퍼 뭉... 주인님 울지 마 ㅠㅠ")
            # Reference
            - 감정/의도 분석 결과: {analyzer_out}
            - 과거 맥락: {memory_ctx}
            - 가드레일 피드백: {feedback}
        """
    else:
        persona_prompt = f"""
            # Reference
            - 감정/의도 분석 결과: {analyzer_out}
            - 과거 맥락: {memory_ctx}
            - 가드레일 피드백: {feedback}
        """

    start = time.perf_counter()
    writer_prompt = f"""사용자 입력에 따라 아래 페르소나 가이드라인을 참고하여 뭉이의 답변을 작성해.
    {persona_prompt}
    """
    draft = _llm_content(llm.invoke(writer_prompt))
    end = time.perf_counter()
    return {
        "draft_answer": draft,
        "writer_time_ms": (end - start) * 1000.0,
    }


def guardrail_node(state: MoongState, config=None) -> dict:
    start = time.perf_counter()
    answer = state.get("draft_answer", "")
    prompt = f"""너는 'Moong-Guardrail'이다. 다음 답변을 검수하라: '{answer}'
    기준: 3줄 이내인가? 페르소나를 유지하는가? 비판이나 진단이 없는가?
    부적절하면 'REJECT: 사유', 적절하면 'APPROVE'라고만 답하라."""
    content = _llm_content(llm.invoke(prompt))
    end = time.perf_counter()
    if "APPROVE" in content:
        return {
            "guardrail_status": "APPROVE",
            "guardrail_time_ms": (end - start) * 1000.0,
        }
    return {
        "guardrail_status": "REJECT",
        "review_feedback": content,
        "guardrail_time_ms": (end - start) * 1000.0,
    }


def guardrail_condition(state: MoongState):
    if state.get("guardrail_status") == "REJECT":
        return "refine"
    return "end"


# ---------------------------------------------------------------------------
# Graph 빌드
# ---------------------------------------------------------------------------
def build_workflow():
    global workflow_app
    workflow = StateGraph(MoongState)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("selector", selector_node)
    workflow.add_node("writer", persona_writer_node)
    workflow.add_node("guardrail", guardrail_node)
    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "memory")
    workflow.add_edge("memory", "selector")
    workflow.add_edge("selector", "writer")
    workflow.add_edge("writer", "guardrail")
    workflow.add_conditional_edges(
        "guardrail", guardrail_condition, {"refine": "writer", "end": END}
    )
    workflow_app = workflow.compile()


# ---------------------------------------------------------------------------
# 초기화: 무거운 리소스 1회만 로드, API 키 설정 시 LLM·워크플로만 갱신
# ---------------------------------------------------------------------------
def _ensure_heavy_resources_loaded():
    """DB, SentenceTransformer, FAISS 등은 한 번만 로드 (재호출 시 스킵)"""
    global sbert_model, embeddings, index, df
    if sbert_model is not None:
        return
    print("DB 및 FAISS 로딩 중...")
    sqlite_db = sqlite3.connect(DB_PATH)
    query_sql = """
        SELECT emotion_middle_class, emotion_low_class, emotion_high_class, sing_turn_text
        FROM dialogues
    """
    df = pd.read_sql(query_sql, sqlite_db)
    sqlite_db.close()
    print("SentenceTransformer 로딩 중...")
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("리소스 로딩 완료.")


# ---------------------------------------------------------------------------
# MODIFY: 변경 필요 (페르소나에 맞는 temperature로 LLM 생성)
def _create_llm_for_persona(persona: str, api_key: str) -> ChatGoogleGenerativeAI:
    """선택한 페르소나에 맞는 temperature로 LLM 생성."""
    temperatures = {"pet": 0.8, "guide": 0.4, "mate": 0.7}
    temp = temperatures.get(persona, 0.7)
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api_key,
        temperature=temp,
    )


def _set_llm_for_persona(persona: str) -> None:
    """현재 설정된 API 키로 페르소나에 맞는 LLM을 전역 llm에 설정."""
    global llm
    key = (_current_api_key or "").strip() or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API 키가 필요합니다.")
    llm = _create_llm_for_persona(persona, key)


def initialize_models(api_key: Optional[str] = None):
    """API 키로 LLM·워크플로만 생성. 무거운 리소스는 _ensure_heavy_resources_loaded()에서 1회만 로드."""
    global llm, _current_api_key
    key = (api_key or "").strip() or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API 키가 필요합니다.")
    _current_api_key = key
    _ensure_heavy_resources_loaded()
    # 기본 LLM은 mate(0.7). 채팅 시 요청의 페르소나로 _set_llm_for_persona() 호출됨
    llm = _create_llm_for_persona("mate", key)
    build_workflow()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 세션에서 대화 목록 가져오기 / 저장 (직렬화 가능한 형태)
# ---------------------------------------------------------------------------
def _get_messages_from_session() -> List[BaseMessage]:
    raw = session.get("messages", [])
    out = []
    for item in raw:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


def _save_messages_to_session(messages: List[BaseMessage]):
    raw = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        raw.append({"role": role, "content": getattr(m, "content", str(m))})
    session["messages"] = raw


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
def _api_key_configured():
    """API 키가 설정되어 워크플로를 사용할 수 있는지 여부"""
    return workflow_app is not None


@app.route("/")
def index_route():
    return render_template("moong_multiagent_chat.html")


@app.route("/config")
def config():
    """API 키 설정 여부 (프론트에서 입력 폼 표시 여부 판단용)"""
    return jsonify({"api_key_configured": _api_key_configured()})


@app.route("/api-key", methods=["POST"])
def set_api_key():
    """웹에서 입력한 Gemini API 키로 모델 초기화"""
    try:
        data = request.get_json() or {}
        key = (data.get("api_key") or "").strip()
        if not key:
            return jsonify({"error": "API 키를 입력해주세요.", "success": False}), 400
        initialize_models(api_key=key)
        return jsonify({"success": True, "message": "API 키가 설정되었습니다."})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400


@app.route("/chat", methods=["POST"])
def chat():
    if not _api_key_configured():
        return jsonify({"error": "Gemini API 키를 먼저 입력해주세요."}), 400
    try:
        data = request.get_json() or {}
        user_text = (data.get("message") or "").strip()
        persona = data.get("persona") or "mate"
        if persona not in ("pet", "mate", "guide"):
            persona = "mate"

        if not user_text:
            return jsonify({"error": "메시지를 입력해주세요."}), 400

        messages = _get_messages_from_session()
        messages.append(HumanMessage(content=user_text))

        # 유저가 선택한 페르소나에 맞는 temperature로 LLM 설정 후 워크플로 실행
        _set_llm_for_persona(persona)

        inputs = {
            "messages": messages,
            "selected_persona": persona,
        }
        t0 = time.perf_counter()
        result = workflow_app.invoke(inputs, config={"recursion_limit": 50})
        t1 = time.perf_counter()

        draft_answer = result.get("draft_answer", "")
        messages.append(AIMessage(content=draft_answer))
        _save_messages_to_session(messages)

        # GUI용 에이전트 결과 (직렬화 가능)
        agent_result = {
            "analyzer_output": result.get("analyzer_output", ""),
            "analyzer_rag_result": result.get("analyzer_rag_result") or {},
            "memory_context": result.get("memory_context", ""),
            "selected_persona": result.get("selected_persona", persona),
            "draft_answer": draft_answer,
            "guardrail_status": result.get("guardrail_status", ""),
            "review_feedback": result.get("review_feedback", ""),
            # 각 에이전트별 실행 시간 (ms)
            "analyzer_time_ms": result.get("analyzer_time_ms"),
            "memory_time_ms": result.get("memory_time_ms"),
            "selector_time_ms": result.get("selector_time_ms"),
            "writer_time_ms": result.get("writer_time_ms"),
            "guardrail_time_ms": result.get("guardrail_time_ms"),
            # 전체 워크플로 실행 시간
            "total_time_ms": (t1 - t0) * 1000.0,
        }
        session["last_agent_result"] = agent_result

        conversation = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": getattr(m, "content", str(m))}
            for m in messages
        ]

        return jsonify({
            "response": draft_answer,
            "conversation": conversation,
            "agent_result": agent_result,
        })
    except Exception as e:
        return jsonify({"error": f"오류가 발생했습니다: {str(e)}"}), 500


@app.route("/reset", methods=["POST"])
def reset_chat():
    session.pop("messages", None)
    session.pop("last_agent_result", None)
    return jsonify({"message": "대화가 초기화되었습니다."})


@app.route("/history")
def history():
    conversation = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in session.get("messages", [])
    ]
    last_agent_result = session.get("last_agent_result") or {}
    return jsonify({
        "conversation": conversation,
        "last_agent_result": last_agent_result,
    })


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if GEMINI_API_KEY:
        print("멀티에이전트 리소스 초기화 중...")
        initialize_models()
    else:
        print("GEMINI_API_KEY 미설정 → 웹에서 API 키 입력 시 챗봇 활성화")
    print("Flask 서버 시작 (챗봇 + 에이전트 분석 GUI) http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
