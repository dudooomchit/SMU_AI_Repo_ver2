# ============================================================
# lv 1 - 환경변수 / 라이브러리
# ============================================================

import os
from dotenv import load_dotenv

import fitz

from collections import Counter

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)


load_dotenv()


# ------------------------------------------------------------
# API Key 확인
# ------------------------------------------------------------

if os.environ.get("OPENAI_API_KEY"):
    print("✓ OpenAI API Key가 설정되었습니다.")
else:
    print("✗ OpenAI API Key가 없습니다.")


# ------------------------------------------------------------
# Qdrant Cloud 설정 확인
# ------------------------------------------------------------

if os.environ.get("QDRANT_URL") and os.environ.get("QDRANT_API_KEY"):
    print("✓ Qdrant Cloud 설정이 완료되었습니다.")
    print(f"  URL: {os.environ.get('QDRANT_URL')}")
else:
    print("✗ Qdrant Cloud 설정이 필요합니다.")
    print("  .env 파일에 QDRANT_URL과 QDRANT_API_KEY를 추가하세요.")


# ============================================================
# lv 2 - 페이지별 카테고리 분류
# ============================================================

def get_category(page_num):
    """
    2026_freshman_c.pdf의 PDF 물리 페이지 번호 기준 분류.

    PDF 뷰어에 표시되는 1~116페이지를 사용한다.
    한 페이지에 여러 카테고리가 있으면
    '복합__카테고리1__카테고리2' 형태로 반환한다.
    """

    # 표지, 학교 소개, 애플리케이션, 목차 등
    if 1 <= page_num <= 9:
        return "기타"

    # --------------------------------------------------------
    # 교무팀 (10 ~ 22)
    # --------------------------------------------------------
    elif 10 <= page_num <= 22:

        if 10 <= page_num <= 16:
            return "졸업_및_수강신청_안내"

        elif 17 <= page_num <= 18:
            return "출결제도_안내"

        elif page_num == 19:
            return "학사제도_안내"

        elif 20 <= page_num <= 21:
            return "학적변동_및_각종_증명서_발급_안내"

        elif page_num == 22:
            return "학부_과_전공_사무실_전화번호_안내"

        else:
            return "교무팀"


    # --------------------------------------------------------
    # 교육혁신추진팀
    # --------------------------------------------------------
    elif 23 <= page_num <= 24:

        if page_num == 23:
            return (
                "복합__SM_IN_핵심역량"
                "__SM_IN_핵심역량_진단_및_인증"
            )

        elif page_num == 24:
            return "학생_참여_프로그램"

        else:
            return "교육혁신추진팀"


    # --------------------------------------------------------
    # 자유전공학부지원센터
    # --------------------------------------------------------
    elif 25 <= page_num <= 27:

        if page_num == 25:
            return (
                "복합__전공탐색_교과목"
                "__교원_학생끌어주기_프로그램"
                "__선후배_이어주기_프로그램"
                "__전공선택_징검다리"
            )

        elif page_num == 26:
            return (
                "자유전공생_전용_공간_"
                "코워킹스페이스_Coworking_Space"
            )

        elif page_num == 27:
            return "자유전공학부생_교육과정"

        else:
            return "자유전공학부지원센터"


    # --------------------------------------------------------
    # 비교과통합지원센터
    # --------------------------------------------------------
    elif 28 <= page_num <= 29:
        return "비교과교육과정"


    # --------------------------------------------------------
    # 지능형로봇사업팀
    # --------------------------------------------------------
    elif 30 <= page_num <= 33:
        return "지능형로봇_혁신융합대학"


    # --------------------------------------------------------
    # 계당교양교육원
    # --------------------------------------------------------
    elif 34 <= page_num <= 37:

        if 34 <= page_num <= 35:
            return (
                "2026학년도_신입생_적용_"
                "교양_교육과정_이수_원칙"
            )

        elif 36 <= page_num <= 37:
            return (
                "계당교양교육원_"
                "의사소통능력개발센터_"
                "비교과_프로그램_안내"
            )

        else:
            return "계당교양교육원"


    # --------------------------------------------------------
    # 학생복지팀
    # --------------------------------------------------------
    elif 38 <= page_num <= 45:

        if page_num == 38:
            return (
                "복합__학생증_발급"
                "__복지시설_현황"
            )

        elif page_num == 39:
            return (
                "복합__통학버스_및_무료_셔틀버스_운행_안내"
                "__학생_대상_안전_보험_안내"
            )

        elif 40 <= page_num <= 45:
            return "장학금_학자금대출_제도_안내"

        else:
            return "학생복지팀"


    # --------------------------------------------------------
    # 장애학생지원센터
    # --------------------------------------------------------
    elif page_num == 46:
        return "장애학생_지원_제도"


    # --------------------------------------------------------
    # 상명소셜임팩트센터
    # --------------------------------------------------------
    elif 47 <= page_num <= 48:
        return "기타"


    # --------------------------------------------------------
    # 취업진로지원팀
    # --------------------------------------------------------
    elif 49 <= page_num <= 50:
        return (
            "취업능력_향상_교육_훈련_및_"
            "진로_설정_프로그램_운영"
        )


    # --------------------------------------------------------
    # 현장실습·일경험지원팀
    # --------------------------------------------------------
    elif 51 <= page_num <= 52:
        return "현장_교육_프로그램"


    # --------------------------------------------------------
    # 학술정보관
    # --------------------------------------------------------
    elif 53 <= page_num <= 61:

        if 53 <= page_num <= 55:
            return "시설안내"

        elif page_num == 56:
            return "자료대출_및_반납"

        elif 57 <= page_num <= 59:
            return "이용자_서비스"

        elif page_num == 60:
            return (
                "복합__e_Book_전자잡지"
                "__e_Learning"
            )

        elif page_num == 61:
            return "학술정보관_모바일_어플리케이션"

        else:
            return "학술정보관"


    # --------------------------------------------------------
    # 대외협력팀
    # --------------------------------------------------------
    elif 62 <= page_num <= 63:
        return "국제교류프로그램_안내"


    # --------------------------------------------------------
    # 교수학습혁신센터
    # --------------------------------------------------------
    elif 64 <= page_num <= 65:
        return "주요업무"


    # --------------------------------------------------------
    # 정보통신지원팀
    # --------------------------------------------------------
    elif 66 <= page_num <= 73:

        if page_num == 66:
            return "공용_컴퓨터_실습실_현황"

        elif 67 <= page_num <= 69:
            return (
                "샘물_포탈서비스_접속_및_"
                "학생메일_Office365_설치"
            )

        elif page_num == 70:
            return "무선랜_인증_방법_및_이용_안내"

        elif page_num == 71:
            return (
                "안드로이드_SANGMYUNG_"
                "무선랜_네트워크_설정_방법"
            )

        elif page_num == 72:
            return (
                "iPhone_SANGMYUNG_"
                "무선랜_네트워크_설정_방법"
            )

        elif page_num == 73:
            return "불법_소프트웨어_사용_금지_안내"

        else:
            return "정보통신지원팀"


    # --------------------------------------------------------
    # 대학원교학팀
    # --------------------------------------------------------
    elif 74 <= page_num <= 78:

        if page_num == 74:
            return "일반대학원_서울"

        elif page_num == 75:
            return "일반대학원_천안"

        elif page_num == 76:
            return "특수대학원_서울"

        elif 77 <= page_num <= 78:
            return "학_석사연계과정_지원안내"

        else:
            return "대학원교학팀"


    # --------------------------------------------------------
    # 학생상담센터
    # --------------------------------------------------------
    elif 79 <= page_num <= 81:

        if page_num == 79:
            return "학생상담센터_주요_프로그램"

        elif 80 <= page_num <= 81:
            return "상담_신청절차_및_이용_방법"

        else:
            return "학생상담센터"


    # --------------------------------------------------------
    # 인권센터
    # --------------------------------------------------------
    elif 82 <= page_num <= 83:

        if page_num == 82:
            return "상담_및_사건처리_절차"

        elif page_num == 83:
            return (
                "복합__온라인_폭력예방통합교육_이수_안내"
                "__인권센터_이용안내"
            )

        else:
            return "인권센터"


    # --------------------------------------------------------
    # 학생생활관
    # --------------------------------------------------------
    elif 84 <= page_num <= 85:
        return "학생생활관_현황"


    # --------------------------------------------------------
    # 상명수련원
    # --------------------------------------------------------
    elif 86 <= page_num <= 96:

        if page_num == 86:
            return "신입생_모집안내"

        elif 87 <= page_num <= 89:
            return "사용안내"

        elif 90 <= page_num <= 91:
            return "본관_시설"

        elif page_num == 92:
            return (
                "복합__체육관_및_다목적강당_시설"
                "__야외_시설"
            )

        elif page_num == 93:
            return (
                "복합__여가_시설"
                "__부엉이박물관"
            )

        elif 94 <= page_num <= 96:
            return "상운관"

        else:
            return "상명수련원"


    # --------------------------------------------------------
    # 예비군대대
    # --------------------------------------------------------
    elif 97 <= page_num <= 105:

        if page_num == 97:
            return "병역판정검사_입영_및_연기"

        elif 98 <= page_num <= 100:
            return "대학직장예비군_편성_및_교육훈련"

        elif 101 <= page_num <= 103:
            return "민방위_경계_공습_경보시_행동요령"

        elif page_num == 104:
            return "화재발생시_행동요령"

        elif page_num == 105:
            return (
                "복합__공연관람시_행동요령"
                "__낙뢰시_행동요령"
                "__태풍시_행동요령"
            )

        else:
            return "예비군대대"


    # --------------------------------------------------------
    # 학생군사교육단 ROTC
    # --------------------------------------------------------
    elif 106 <= page_num <= 108:
        return "학군사관_후보생_선발_교육"


    # --------------------------------------------------------
    # 상명스포츠센터
    # --------------------------------------------------------
    elif 109 <= page_num <= 110:

        if page_num == 109:
            return (
                "복합__시설현황"
                "__운영_안내"
            )

        elif page_num == 110:
            return (
                "복합__기타_운영사항"
                "__특이사항"
            )

        else:
            return "상명스포츠센터"


    # --------------------------------------------------------
    # 총무인사회계팀
    # --------------------------------------------------------
    elif page_num == 111:
        return (
            "복합__건물_출입문_개폐시간_안내"
            "__상명대학교_전동_킥보드_및_자전거_PM_"
            "운행_및_주차_구역_안내도"
        )


    # --------------------------------------------------------
    # 우편취급국
    # --------------------------------------------------------
    elif page_num == 112:
        return "우편취급국_이용_안내"


    # --------------------------------------------------------
    # 보건건강관리센터
    # --------------------------------------------------------
    elif page_num == 113:
        return "보건건강관리센터_이용_안내"


    # --------------------------------------------------------
    # 교가, 캠퍼스 배치도
    # --------------------------------------------------------
    elif 114 <= page_num <= 115:
        return "기타"


    # --------------------------------------------------------
    # 마지막 ROTC 페이지
    # --------------------------------------------------------
    elif page_num == 116:
        return "학군사관_후보생_선발_교육"


    else:
        return "기타"


# ============================================================
# lv 3 - PDF 읽기 → LangChain Document 생성
# ============================================================

# 반드시 실제 PDF 파일명 확인할 것
file_path = "../datasets/2026_freshman_c.pdf"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"PDF 파일을 찾을 수 없습니다: {file_path}"
    )


pdf = fitz.open(file_path)

docs = []

print(f"\nPDF 페이지 수: {len(pdf)}")


for page_index in range(len(pdf)):

    page = pdf[page_index]

    # PyMuPDF는 0부터 시작하지만,
    # 우리가 만든 분류표는 PDF 1페이지부터 시작함.
    page_num = page_index + 1

    # 텍스트 추출
    text = page.get_text(
        "text",
        sort=True
    ).strip()

    # 텍스트가 없는 페이지는 제외
    if not text:
        print(
            f"⚠ PDF {page_num}페이지: "
            "텍스트 없음 → 건너뜀"
        )
        continue

    category = get_category(page_num)


    # --------------------------------------------------------
    # 복합 카테고리 분해
    # --------------------------------------------------------

    if category.startswith("복합__"):

        category_list = (
            category
            .replace("복합__", "", 1)
            .split("__")
        )

    else:

        category_list = [category]


    # --------------------------------------------------------
    # LangChain Document 생성
    # --------------------------------------------------------

    document = Document(
        page_content=text,
        metadata={
            "source": os.path.basename(file_path),
            "page": page_num,

            # 원본 카테고리
            "category": category,

            # 필터링용 개별 카테고리 배열
            "categories": category_list,

            "year": 2026,
        }
    )

    docs.append(document)


pdf.close()


print(f"\n총 {len(docs)}개의 문서 생성 완료")


if docs:

    print("\n첫 번째 문서 메타데이터:")

    for key, value in docs[0].metadata.items():
        print(f"  - {key}: {value}")

    print("\n첫 번째 문서 내용 일부:")
    print(docs[0].page_content[:500])

else:

    raise RuntimeError(
        "PDF에서 생성된 Document가 없습니다."
    )


# ============================================================
# lv 4 - 카테고리 검수
# ============================================================

categories = [
    doc.metadata["category"]
    for doc in docs
]

category_counts = Counter(categories)


print("\n카테고리별 문서 수:")

for category, count in sorted(category_counts.items()):
    print(
        f"  - {category}: "
        f"{count}개"
    )


print(
    f"\n전체 Document 수: {len(docs)}개"
)

print(
    f"전체 카테고리 종류: "
    f"{len(category_counts)}개"
)


# ------------------------------------------------------------
# 페이지별 분류 결과 확인
# ------------------------------------------------------------

print("\n페이지별 카테고리:")

for document in docs:

    print(
        f"PDF {document.metadata['page']:3d} "
        f"→ {document.metadata['category']}"
    )


# ============================================================
# lv 5 - Qdrant Cloud 연결
# ============================================================

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

print("\n✓ Qdrant Cloud에 연결되었습니다.")


# ============================================================
# lv 6 - Embedding / Collection 생성
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


# 천안 프로젝트와 완전히 분리
collection_name = "freshman_2026_metadata"


# 기존 컬렉션 존재 여부
collections = client.get_collections().collections

existing_collection = any(
    collection.name == collection_name
    for collection in collections
)


# 새 데이터를 넣을지 결정
should_ingest = True


if existing_collection:

    print(
        f"\n컬렉션 "
        f"'{collection_name}'이 이미 존재합니다."
    )

    user_input = input(
        "기존 데이터를 삭제하고 "
        "새로 추가하시겠습니까? (y/n): "
    )


    if user_input.lower() == "y":

        print(
            f"컬렉션 "
            f"'{collection_name}' 삭제 중..."
        )

        client.delete_collection(
            collection_name=collection_name
        )

        print("✓ 기존 컬렉션 삭제 완료")


    else:

        print(
            "기존 컬렉션을 그대로 사용합니다."
        )

        # 중복 데이터 삽입 방지
        should_ingest = False


# ------------------------------------------------------------
# 컬렉션 생성
# ------------------------------------------------------------

if not existing_collection or should_ingest:

    # 기존 컬렉션을 삭제했다면 새로 생성
    if should_ingest:

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE
            )
        )

        print(
            f"✓ 컬렉션 "
            f"'{collection_name}' 생성 완료"
        )


        # ----------------------------------------------------
        # Metadata Payload Index
        # ----------------------------------------------------

        print(
            "메타데이터 필드 인덱스 생성 중..."
        )


        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.category",
            field_schema=PayloadSchemaType.KEYWORD
        )


        # 복합 카테고리 검색용
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.categories",
            field_schema=PayloadSchemaType.KEYWORD
        )


        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.page",
            field_schema=PayloadSchemaType.INTEGER
        )


        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema=PayloadSchemaType.KEYWORD
        )


        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.year",
            field_schema=PayloadSchemaType.INTEGER
        )


        print("✓ 인덱스 생성 완료")


# ============================================================
# lv 7 - Vector Store 생성 + 데이터 저장
# ============================================================

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)


if should_ingest:

    print(
        f"\n{len(docs)}개 문서 임베딩 및 "
        "Qdrant 저장 시작..."
    )

    vectorstore.add_documents(
        documents=docs
    )

    print(
        f"✓ {len(docs)}개의 문서가 "
        "Qdrant Cloud에 추가되었습니다."
    )

else:

    print(
        "\n기존 Qdrant 데이터를 그대로 사용합니다."
    )


# ============================================================
# lv 8 - Retriever 테스트
# ============================================================

# 테스트할 카테고리
search_category = "학생증_발급"

# 테스트 질문
search_query = "학생증은 어떻게 발급받나요?"


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,

        # category가 아니라 categories 사용
        # 복합 카테고리 안에서도 검색 가능
        "filter": models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.categories",
                    match=models.MatchValue(
                        value=search_category
                    )
                )
            ]
        )
    }
)


results = retriever.invoke(
    search_query
)


print(
    f"\nRetriever 검색 결과 "
    f"[{search_category}] "
    f"{len(results)}개\n"
)


for i, result in enumerate(
    results,
    start=1
):

    print(
        f"{i}. "
        f"PDF {result.metadata['page']}페이지"
    )

    print(
        f"   category: "
        f"{result.metadata['category']}"
    )

    print(
        f"   categories: "
        f"{result.metadata.get('categories')}"
    )

    print()

    print(
        result.page_content[:1000]
    )

    print(
        "\n"
        + "=" * 80
        + "\n"
    )


# ============================================================
# lv 9 - End
# ============================================================

print("✓ 전체 작업 완료")
