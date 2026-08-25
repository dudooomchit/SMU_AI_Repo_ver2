import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# 경로 설정
# ============================================================

# 현재 파일 위치:
# src/demo/streamlit_example.py

CURRENT_DIR = Path(__file__).resolve().parent

# src 디렉토리
SRC_DIR = CURRENT_DIR.parent

# static 이미지 경로
PROP_PATH = CURRENT_DIR / "static" / "Prop.png"
LOGO_PATH = CURRENT_DIR / "static" / "Logo.png"

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(SRC_DIR))


# ============================================================
# AI Graph
# ============================================================

from ai import create_graph


# ============================================================
# 환경 변수 로드
# ============================================================

load_dotenv()


# ============================================================
# LangGraph 생성
# ============================================================

graph = create_graph()


# ============================================================
# CSS 로드
# ============================================================

def load_css():
    css_path = CURRENT_DIR / "style.css"

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# 세션 상태 초기화
# ============================================================

def init_session_state():
    """세션 상태 초기화"""

    if "messages" not in st.session_state:
        st.session_state.messages = []


# ============================================================
# 메시지 표시
# ============================================================

def display_message(
    role: str,
    content: str,
    workflow_info: dict = None
):
    """메시지 표시"""

    # AI 메시지
    if role == "assistant":

        with st.chat_message(
            "assistant",
            avatar=str(PROP_PATH)
        ):
            st.markdown(content)

            if workflow_info:
                display_workflow_info(workflow_info)

    # 사용자 메시지
    else:

        with st.chat_message(role):
            st.markdown(content)


# ============================================================
# 워크플로 정보 표시
# ============================================================

def display_workflow_info(result: dict):
    """워크플로 정보 표시"""

    with st.expander("🔍 워크플로 정보"):

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # 의도 / 재시도
        # ----------------------------------------------------

        with col1:

            st.metric(
                "의도",
                result.get("intent", "N/A")
            )

            if result.get("retry_count"):

                st.metric(
                    "재시도 횟수",
                    result["retry_count"]
                )

        # ----------------------------------------------------
        # 검색 결과
        # ----------------------------------------------------

        with col2:

            if result.get("vector_results"):

                st.metric(
                    "검색된 문서",
                    len(result["vector_results"])
                )

            if result.get("db_results"):

                st.info(
                    "DB 검색 수행됨"
                )

        # ----------------------------------------------------
        # 벡터 검색 결과
        # ----------------------------------------------------

        if result.get("vector_results"):

            st.markdown(
                "#### 📄 검색된 문서"
            )

            for i, doc in enumerate(
                result["vector_results"],
                1
            ):

                source = doc.metadata.get(
                    "source",
                    "알 수 없음"
                )

                with st.expander(
                    f"문서 {i}: {source}"
                ):

                    meta_cols = st.columns(3)

                    # 페이지
                    with meta_cols[0]:

                        st.caption(
                            f"📖 페이지: "
                            f"{doc.metadata.get('page', 'N/A')}"
                        )

                    # 카테고리
                    with meta_cols[1]:

                        category = doc.metadata.get(
                            "category"
                        )

                        if category:

                            st.caption(
                                f"🏷️ 카테고리: {category}"
                            )

                    # 점수
                    with meta_cols[2]:

                        score = doc.metadata.get(
                            "score"
                        )

                        if score is not None:

                            st.caption(
                                f"⭐ 점수: {score:.3f}"
                            )

                    # 내용
                    st.markdown(
                        "**내용:**"
                    )

                    content = doc.page_content

                    st.text(
                        content[:500]
                        + (
                            "..."
                            if len(content) > 500
                            else ""
                        )
                    )

        # ----------------------------------------------------
        # SQL 쿼리
        # ----------------------------------------------------

        if result.get("sql_query"):

            st.markdown(
                "#### SQL 쿼리"
            )

            st.code(
                result["sql_query"],
                language="sql"
            )

        # ----------------------------------------------------
        # 재작성된 쿼리
        # ----------------------------------------------------

        if result.get("rewritten_query"):

            st.info(
                f"재작성된 쿼리: "
                f"{result['rewritten_query']}"
            )

        # ----------------------------------------------------
        # 오류
        # ----------------------------------------------------

        if result.get("error"):

            st.error(
                f"오류: {result['error']}"
            )


# ============================================================
# 메인
# ============================================================

def main():
    """메인 함수"""

    # --------------------------------------------------------
    # 페이지 설정
    # --------------------------------------------------------

    st.set_page_config(
        page_title="상명대학교 AI 에이전트",
        page_icon="🎓",
        layout="wide"
    )

    # CSS 적용
    load_css()

    # --------------------------------------------------------
    # 제목
    # --------------------------------------------------------

    st.title(
        "🎓 상명대학교 AI 에이전트"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 세션 초기화
    # --------------------------------------------------------

    init_session_state()

    # ========================================================
    # 사이드바
    # ========================================================

    with st.sidebar:

        # ----------------------------------------------------
        # 상명대학교 로고
        # ----------------------------------------------------

        st.image(
            str(LOGO_PATH),
            width=170
        )

        # ----------------------------------------------------
        # 설정 확인
        # ----------------------------------------------------

        st.markdown(
            "## ⚙ 설정 확인"
        )

        required_vars = {

            "OPENAI_API_KEY":
                "OpenAI API",

            "QDRANT_URL":
                "Qdrant URL",

            "QDRANT_API_KEY":
                "Qdrant API Key",

            "SUPABASE_DB_URL":
                "Supabase DB"
        }

        for var, name in required_vars.items():

            if os.getenv(var):

                st.success(
                    f"✓ {name}"
                )

            else:

                st.error(
                    f"✗ {name}"
                )

        st.markdown("---")

        # ----------------------------------------------------
        # 사용 방법
        # ----------------------------------------------------

        st.header(
            "📖 사용 방법"
        )

        st.markdown(
            """
**일반 질문**
- 안녕하세요
- 고마워

**문서 검색**
- 교내 장학금 종류는?
- 휴학 신청 시 필요한 서류는?
- 다전공이란 무엇인가요?
- 교내 동아리 활동에 대해 알려주세요.

**DB 검색**
- 한식 업종의 행정동은?
- 천안시 착한가격업소 중 한식 업종은 몇 개인가요?
- 청룡각의 소재지 주소는?
"""
        )

        # ----------------------------------------------------
        # 대화 초기화
        # ----------------------------------------------------

        if st.button(
            "대화 초기화",
            type="secondary"
        ):

            st.session_state.messages = []

            st.rerun()

    # ========================================================
    # 이전 대화 출력
    # ========================================================

    for message in st.session_state.messages:

        display_message(
            message["role"],
            message["content"],
            message.get(
                "workflow_info"
            )
        )

    # ========================================================
    # 사용자 입력
    # ========================================================

    prompt = st.chat_input(
        "질문을 입력하세요..."
    )

    # ========================================================
    # 질문 처리
    # ========================================================

    if prompt:

        # ----------------------------------------------------
        # 사용자 메시지 표시
        # ----------------------------------------------------

        display_message(
            "user",
            prompt
        )

        # 세션 저장
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # ====================================================
        # AI 답변
        # Prop.png 프로필 적용
        # ====================================================

        with st.chat_message(
            "assistant",
            avatar=str(PROP_PATH)
        ):

            with st.spinner(
                "생각 중..."
            ):

                try:

                    # ----------------------------------------
                    # 현재 대화 전체를
                    # LangGraph messages 형식으로 변환
                    # ----------------------------------------

                    graph_messages = []

                    for message in st.session_state.messages:

                        if message["role"] in [
                            "user",
                            "assistant"
                        ]:

                            graph_messages.append(
                                {
                                    "role":
                                        message["role"],

                                    "content":
                                        message["content"]
                                }
                            )

                    # ----------------------------------------
                    # LangGraph 실행
                    # ----------------------------------------

                    result = graph.invoke(
                        {
                            "messages":
                                graph_messages
                        }
                    )

                    # ----------------------------------------
                    # 결과 메시지 가져오기
                    # ----------------------------------------

                    messages = result.get(
                        "messages",
                        []
                    )

                    # ----------------------------------------
                    # 마지막 AI 답변 추출
                    # ----------------------------------------

                    if messages:

                        last_message = messages[-1]

                        if hasattr(
                            last_message,
                            "content"
                        ):

                            answer = (
                                last_message.content
                            )

                        elif isinstance(
                            last_message,
                            dict
                        ):

                            answer = (
                                last_message.get(
                                    "content",
                                    ""
                                )
                            )

                        else:

                            answer = str(
                                last_message
                            )

                    else:

                        answer = (
                            "죄송합니다. "
                            "답변을 생성할 수 없습니다."
                        )

                    # ----------------------------------------
                    # AI 답변 출력
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )

                    # ----------------------------------------
                    # 워크플로 정보 출력
                    # ----------------------------------------

                    display_workflow_info(
                        result
                    )

                    # ----------------------------------------
                    # AI 답변 세션 저장
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                answer,

                            "workflow_info":
                                result
                        }
                    )

                # =================================================
                # 오류 처리
                # =================================================

                except Exception as e:

                    error_msg = (
                        f"오류가 발생했습니다: "
                        f"{str(e)}"
                    )

                    st.error(
                        error_msg
                    )

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                error_msg
                        }
                    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()