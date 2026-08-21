from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ai.state import AgentState
from ai.retriever import get_retriever
from ai.text2sql import get_text2sql_engine
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

llm = init_chat_model("gpt-5.4-mini")

_retriever = None
_text2sql_engine = None


class VectorSearchQuery(BaseModel):
    """벡터 검색을 위한 쿼리 분석 결과"""
    optimized_query: str = Field(
        description="검색에 최적화된 쿼리. 핵심 키워드를 포함하고 명확하게 작성."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="선택된 카테고리 리스트 (1-2개). 명확하게 관련 있는 카테고리만 선택. 애매하거나 불확실한 경우 null 반환. 가능한 값: 새로운_천안_신규_조성_공간, 일상편의_행정_교통, 아이돌봄, 맞춤_복지, 건강_내일_보건, 문화_관광, 즐거운_일상_행사축제, 행복한_변화_달라지는_정책"
    )


def get_cached_retriever():
    """캐시된 retriever 인스턴스 반환 (lazy initialization)"""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def get_cached_text2sql_engine():
    """캐시된 text2sql_engine 인스턴스 반환 (lazy initialization)"""
    global _text2sql_engine
    if _text2sql_engine is None:
        _text2sql_engine = get_text2sql_engine()
    return _text2sql_engine


def classify_intent(state: AgentState) -> AgentState:
    """
    사용자 질문의 의도를 분류하는 노드

    분류 결과:
    - 'general': 일반적인 대화나 인사
    - 'database': 데이터베이스 조회가 필요한 질문
    - 'vector': 문서 검색이 필요한 질문

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태.
    """
    # messages에서 질문 추출
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided")

    # 마지막 사용자 메시지를 질문으로 사용
    question = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    system_prompt = """
당신은 사용자 질문의 의도를 분류하는 전문가입니다.~

이전 대화 맥락을 고려하여 현재 질문을 다음 3가지 중 하나로 분류하세요:

1. 'general' - 일반적인 대화, 인사, 간단한 질문
   예: "안녕하세요", "고마워", "날씨 어때?"

2. 'database' - 업종, 소재지 주소, 대표자 이름, 업소명 등 조직도 데이터베이스 조회가 필요한 질문
   예: "한식 업종의 행정동은?", "천안시 착한가격 업소 중 한식 업종은 몇개인가요?", "천안시 청룡각의 소재지주소는 무엇인가요?"

3. 'vector' - 출결 제도, 학사제도, 증명서 발급, 장학금 등 문서 검색이 필요한 질문
   예: "교내 장학금 종류의 갯수는?", "휴학 신청시 구비해야 할 서류는 무엇인가요?", "다전공이란 무엇인가요?"

반드시 'general', 'database', 'vector' 중 하나만 답변하세요.
다른 설명 없이 분류 결과만 반환하세요.
"""

    # 시스템 메시지 + 전체 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    intent = response.content.strip().lower()

    # 유효한 의도인지 확인
    if intent not in ['general', 'database', 'vector']:
        intent = 'general'

    return {
        "intent": intent,
        "question": question
    }


def general_answer(state: AgentState) -> AgentState:
    """
    일반적인 질문에 직접 답변하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 자연스럽고 도움이 되는 답변을 제공하세요.
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


def vector_search(state: AgentState) -> AgentState:
    """
    Qdrant 벡터 검색을 수행하는 노드

    1. LLM으로 질문 분석 (최적화된 쿼리 + 카테고리 추출)
    2. 병렬 벡터 검색 수행

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    # 재작성된 쿼리가 있으면 사용, 없으면 원본 질문 사용
    original_query = state.get("rewritten_query") or state.get("question", "")

    # 이전 대화 맥락이 있으면 완전한 질문으로 재구성
    if len(messages) > 1 and not state.get("rewritten_query"):
        # rewritten_query가 없을 때만 (첫 시도) 맥락 고려
        system_prompt_complete = """
당신은 질문 분석 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "서산집 업종은 무엇?" → 현재: "대표자는 누구야?" → 재구성: "서산집의 대표자는 누구야?"
- 이전: "장학금 제도" → 현재: "더 자세히 알려줘" → 재구성: "장학금 제도에 대해 더 자세히 알려줘"

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
만약 현재 질문이 이미 완전하다면 그대로 반환하세요.
"""
        conversation_complete = [SystemMessage(content=system_prompt_complete)] + messages
        response_complete = llm.invoke(conversation_complete)
        original_query = response_complete.content.strip()

    # 1. LLM으로 쿼리 분석 및 카테고리 추출 (Structured Output)
    # 시스템 프롬프트: 역할 정의 및 카테고리 설명
    system_prompt = """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 분석하여 벡터 검색에 최적화된 쿼리를 생성하고, 적절한 카테고리를 선택하는 역할을 수행합니다.

사용 가능한 카테고리:
- 교무팀 (졸업, 수강신청, 학사제도, 성적, 학적변동, 증명서 발급 등) : 졸업 및 수강신청 안내, 출결제도 안내, 학사제도 안내, 학적변동 및 각종 증명서 발급 안내, 학부(과/전공)사무실 전화번호 안내 관련 
- 교육혁신추진팀 (SM-IN 핵심역량 진단 및 인증, 상명피날레) : SM-IN 핵심역량, SM-IN 핵심역량 진단 및 인증, 학생 참여 프로그램 관련
- 자유전공학부지원센터  (전공탐색/전공체험 교과목, 선후배 이어주기 프로그램 등) : 전공탐색 교과목, 교원-학생끌어주기 프로그램, 선후배 이어주기 프로그램, 전공선택 징검다리, 자유전공생 전용 공간 코워킹스페이스(Coworking Space), 자유전공학부생 교육과정 관련
- 비교과통합지원센터(비교과교육과정 운영, 비교과 마일리지 제도 관리, 비교과 프로그램 관련 상담 등) : 비교과교육과정 관련
- 지능형로봇사업팀 (지능형로봇 혁신융합대학 사업 관련 교과·비교과 프로그램 운영) : 지능형로봇 혁신융합대학 관련
- 계당교양교육원 교학지원팀 (교양 교육과정, 비교과 프로그램 운영) : 2026학년도 신입생 적용 교양 교육과정 이수 원칙, 계당교양교육원 의사소통능력개발센터 비교과 프로그램 안내
- 학생복지팀 (학생증 발급, 복지시설, 통학·셔틀버스, 학생 보험, 장학제도 및 학자금 대출): 학생증 발급, 복지시설 현황, 통학버스 및 무료 셔틀버스 운행 안내, 학생 대상 안전 보험 안내, 장학금/학자금대출 제도 안내 관련
- 장애학생지원센터 (장애학생 지원 제도): 장애학생 지원 제도 관련
- 상명소셜임팩트센터(SSIC) (상명사회봉사단, 지역사회공헌 활동): 주요 업무, 상명 사회봉사단,봉사단 서포터즈 '날개단',주요활동(홈페이지 사회봉사 공고 확인 필수) 관련
- 취업진로지원팀 (취업능력 향상 교육·훈련 및 진로 설정 프로그램 운영): 취업능력 향상 교육·훈련 및 진로 설정 프로그램 운영 관련
- 현장실습·일경험지원팀":"현장 교육 프로그램 관련
- 학술정보관 (도서관 이용 안내): 시설안내,자료대출 및 반납,이용자 서비스,e-Book/전자잡지,e-Learning,학술정보관 모바일 어플리케이션 관련
- 대외협력팀 (교환학생, 복수학위, 해외단기연수, 홍보대사, 대학SNS, 대학홍보, 발전기금, 국내외기관 교류 등): 국제교류프로그램 안내 관련
- 교수학습혁신센터 (e-Campus 플랫폼 관리, 학습지원 프로그램(상명튜터링, 스터디상생플러스 등)): "주요업무 및 e-Campus 플랫폼 관리, 학습지원 프로그램 안내 관련


카테고리 선택 규칙:
1. 명확하게 관련 있는 카테고리를 1-2개 선택합니다
2. 여러 카테고리와 관련될 수 있으면 최대 2개까지 선택
3. 애매하거나 확신이 없으면 반드시 null을 반환 (잘못된 카테고리보다 null이 나음)
4. 억지로 카테고리를 선택하지 말고, 확실한 경우에만 선택

출력 지침:
1. optimized_query: 검색에 효과적인 핵심 키워드를 포함한 쿼리로 최적화
2. categories: 명확하게 관련 있는 카테고리 1-2개를 리스트로 반환. 불확실하면 null"""

    # 유저 프롬프트: 실제 질문
    user_prompt = f"다음 질문을 분석해주세요:\n\n{original_query}"

    # 메시지 객체 생성 (Structured Output용)
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Structured Output으로 LLM 호출
    structured_llm = llm.with_structured_output(VectorSearchQuery)
    query_analysis = structured_llm.invoke(llm_messages)

    optimized_query = query_analysis.optimized_query
    categories = query_analysis.categories

    print(f"[벡터 검색 쿼리 분석]")
    print(f"  원본 쿼리: {original_query}")
    print(f"  최적화된 쿼리: {optimized_query}")
    print(f"  선택된 카테고리: {categories}")

    # 2. 병렬 벡터 검색 수행 (카테고리 필터 적용)
    retriever = get_cached_retriever()
    results = retriever.search(optimized_query, k=3, score_threshold=0.5, categories=categories)

    return {
        "vector_results": results
    }


def rewrite_query(state: AgentState) -> AgentState:
    """
    검색 결과가 부족할 때 쿼리를 재작성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 검색 쿼리 최적화 전문가입니다.

사용자의 질문이 검색 결과를 얻지 못했습니다.
이전 대화 맥락을 고려하여 질문을 다시 작성하여 더 나은 검색 결과를 얻을 수 있도록 하세요.

최적화 방법:
- 이전 대화에서 언급된 맥락을 포함
- 동의어나 관련 용어 추가
- 질문을 더 구체적이거나 더 일반적으로 변경
- 핵심 키워드 강조

재작성된 쿼리만 반환하세요. 설명은 포함하지 마세요.
"""

    # 시스템 메시지 + 전체 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    rewritten = response.content.strip()

    return {
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1
    }


def database_query(state: AgentState) -> AgentState:
    """
    Text2SQL을 수행하여 데이터베이스를 조회하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])
    question = state.get("question", "")
    previous_error = state.get("error")

    # 이전 대화 맥락이 있으면 완전한 질문으로 재구성
    if len(messages) > 1:
        system_prompt = """
당신은 질문 분석 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "한식 업종 알려줘" → 현재: "행정동은?" → 재구성: "한식 업종의 행정동은?"
- 이전: "운영식당" → 현재: "거기 주소는?" → 재구성: "운영식당의 주소는?"

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
만약 현재 질문이 이미 완전하다면 그대로 반환하세요.
"""
        conversation = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(conversation)
        complete_question = response.content.strip()
    else:
        complete_question = question

    # Text2SQL 실행
    text2sql_engine = get_cached_text2sql_engine()
    result = text2sql_engine.query(complete_question, previous_error=previous_error)

    return {
        "sql_query": result["sql_query"],
        "db_results": result["result"],
        "error": result["error"],
        "retry_count": state.get("retry_count", 0) + 1
    }


def generate_answer(state: AgentState) -> AgentState:
    """
    검색 결과를 바탕으로 최종 답변을 생성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    # 컨텍스트 구성
    context_parts = []

    # 벡터 검색 결과가 있으면 추가
    if state.get("vector_results"):
        docs = state["vector_results"]
        context_parts.append("관련 문서:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "알 수 없음")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "")

            # 출처 정보 구성
            source_info = f"출처: {source}, 페이지: {page}"
            if category:
                source_info += f", 카테고리: {category}"

            context_parts.append(f"\n[문서 {i}] {source_info}\n{doc.page_content}")

    # DB 검색 결과가 있으면 추가
    if state.get("db_results"):
        context_parts.append(f"\n\n데이터베이스 조회 결과:\n{state['db_results']}")
        if state.get("sql_query"):
            context_parts.append(f"\n실행된 SQL:\n{state['sql_query']}")

    context = "\n".join(context_parts)

    system_prompt = f"""
당신은 천안시 착한가격업소 및 상명대학교 천안캠퍼스의 정보 전문가입니다.

다음 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요:

<context>
{context}
</context>

답변 시 다음 규칙을 따르세요:
- 주어진 정보를 자연스럽고 간결하게 전달하세요
- 구체적인 날짜, 수치, 위치, 전화번호 등의 정보를 명확히 포함하세요
- 불필요한 전제 조건이나 한계를 언급하지 말고, 질문에 직접 답변하세요
- 정보가 정말로 없는 경우에만 "해당 정보를 찾을 수 없습니다"라고 말하세요
- 천안시민 및 학생들에게 도움이 되는 친절하고 자연스러운 어조로 답변하세요
- 이전 대화 맥락을 고려하여 답변하세요
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


def route_by_intent(state: AgentState) -> str:
    """
    의도에 따라 다음 노드를 결정하는 라우팅 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    intent = state.get("intent", "general")

    if intent == "general":
        return "general_answer"
    elif intent == "database":
        return "database_query"
    elif intent == "vector":
        return "vector_search"
    else:
        return "general_answer"


def check_vector_results(state: AgentState) -> str:
    """
    벡터 검색 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    results = state.get("vector_results", [])
    retry_count = state.get("retry_count", 0)

    # 결과가 있거나 재시도 횟수가 2회 이상이면 답변 생성
    retriever = get_cached_retriever()
    if retriever.is_relevant(results) or retry_count >= 2:
        return "generate_answer"
    else:
        return "rewrite_query"


def check_db_results(state: AgentState) -> str:
    """
    데이터베이스 검색 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    error = state.get("error")
    result = state.get("db_results")
    retry_count = state.get("retry_count", 0)

    # 오류가 없고 결과가 있으면 답변 생성
    text2sql_engine = get_cached_text2sql_engine()
    if not error and result and not text2sql_engine.is_empty_result(result):
        return "generate_answer"

    # 재시도 횟수가 2회 이상이면 답변 생성 (오류 메시지 포함)
    if retry_count >= 2:
        return "generate_answer"

    # 재시도
    return "database_query"
