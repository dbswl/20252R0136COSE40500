
!pip -q install keybert sentence-transformers
!pip -q install konlpy

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd
import textwrap, re

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
kw_model = KeyBERT(model=SentenceTransformer(model_name))

doc = textwrap.dedent("""
넷플릭스 애니메이션 영화 케이팝 데몬 헌터스의 글로벌 흥행 여파로 핼러윈 시즌을 앞둔 전 세계 부모들이 캐릭터 의상을 구하는 데 큰 어려움을 겪고 있다고 월스트리트저널이 24일 보도했다. 펜실베이니아주 피츠버그의 교사 켈리 빌닝은 9살 딸 알라나가 핼러윈에 케데헌 캐릭터 조이로 분장하고 싶어 했지만, 온라인에서 구할 수 있는 의상은 거의 없거나 품질이 낮았다. 결국 빌닝은 한 소매점에 마지막 남은 의상을 가까스로 구해 첫 파티에 참석할 수 있었다. 빌닝은 핼러윈을 매년 즐기지만, 이번에는 의상을 구하기 위해 두 배 이상의 돈이 들었다고 말했다. 북미 전역에서 부모들은 케이팝 악마 사냥꾼과 라이벌 사자 보이즈로 아이들을 분장시키기 위해 안간힘을 쓰고 있다. 온라인에서 구할 수 있는 의상은 대부분 품절 상태이거나, 품질이 낮거나, 신뢰하기 어려운 공급처 제품이 많고, 배송 기간도 몇 주로 늘어져 있다. 넷플릭스 공식 온라인 스토어에서도 캐릭터 의상을 판매하고 있으나, 가격 부담이 크다. 주인공 루미가 착용한 노란색 재킷만 89.95달러이며, 여기에 블루 숏팬츠, 전투용 부츠, 허리까지 오는 보라색 가발 등을 모두 갖추려면 상당한 비용이 든다. 이 같은 인기 폭주로 일부 부모들은 재봉틀과 글루건을 이용해 직접 의상을 제작하거나, 점토로 장식을 만들어 캐릭터의 디테일을 살리기도 한다. 케데헌은 넷플릭스가 2021년 소니의 미국 영화 라이선스 계약을 체결하면서 스트리밍용으로 제작한 오리지널 작품 중 하나다. 제작 당시에는 비교적 틈새 팬층만 겨냥한 작품으로 여겨졌다. 그러나 6월 공개 직후 넷플릭스 역대 최고 인기 영화가 되었고, 8월 극장 개봉 시에도 박스오피스 1위를 차지하며 예상치를 훌쩍 뛰어넘는 흥행을 기록했다. 영화의 성공은 넷플릭스의 상품화 계획에도 큰 영향을 미쳤다. 넷플릭스는 영화 공개 전부터 라이선싱 박람회에서 의류·장난감 업체에 상품화를 제안했으나, 케데헌이 새롭고 검증되지 않은 애니메이션이라는 이유로 업체들이 큰 투자를 꺼렸다. 결국 영화가 대히트를 치자 넷플릭스는 핼러윈 시즌을 앞두고 급히 공식 상품과 의상을 준비해야 했지만 역부족이었다. 미국 캘리포니아주 글렌데일에 위치한 핼러윈 의상 체인점 스피릿 핼러윈에서도 케데헌 의상은 출시 직후 품절됐다. 매장 직원은 두 번의 입고가 있었지만, 들어오는 즉시 모두 팔렸다. 부모들에게는 아마존에서 찾아보라고 안내하고 있다고 전했다.
""").strip()

def normalize(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

doc = normalize(doc)
len(doc), doc[:200]

keywords_full = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(1, 3),
    top_n=10,
    use_mmr=False,
)
df_full = pd.DataFrame(keywords_full, columns=["keyword","similarity_full"])\
           .sort_values("similarity_full", ascending=False).reset_index(drop=True)
df_full

mid = len(doc) // 2
doc_left, doc_right = doc[:mid], doc[mid:]

left_kw = kw_model.extract_keywords(
    doc_left, keyphrase_ngram_range=(1,3), top_n=5, use_mmr=False
)
right_kw = kw_model.extract_keywords(
    doc_right, keyphrase_ngram_range=(1,3), top_n=5, use_mmr=False
)

df_left  = pd.DataFrame(left_kw,  columns=["keyword","similarity_left"])\
            .sort_values("similarity_left", ascending=False).reset_index(drop=True)
df_right = pd.DataFrame(right_kw, columns=["keyword","similarity_right"])\
            .sort_values("similarity_right", ascending=False).reset_index(drop=True)

df_left, df_right

set_full = set(df_full["keyword"])
set_left = set(df_left["keyword"])
set_right= set(df_right["keyword"])
set_seg  = set_left | set_right

only_in_segments = sorted(list(set_seg - set_full))
only_in_full     = sorted(list(set_full - set_seg))
in_both          = sorted(list(set_full & set_seg))

only_in_segments, only_in_full, in_both

def score(df, key, col):
    row = df[df["keyword"]==key]
    return float(row[col].values[0]) if len(row) else None

rows=[]
for k in in_both:
    rows.append({
        "keyword": k,
        "full_score":  score(df_full,  k, "similarity_full"),
        "left_score":  score(df_left,  k, "similarity_left"),
        "right_score": score(df_right, k, "similarity_right")
    })
df_overlap = pd.DataFrame(rows).sort_values("full_score", ascending=False).reset_index(drop=True)
df_overlap

from IPython.display import display
import pandas as pd
import math, re

def summarize_keybert_results(df_full, df_left, df_right):

    def canon(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w가-힣\s]", " ", s)   # 문장부호 제거
        s = re.sub(r"\s+", " ", s).strip()
        return s

    set_full = set(df_full["keyword"])
    set_left = set(df_left["keyword"])
    set_right= set(df_right["keyword"])
    set_seg  = set_left | set_right

    only_in_segments = sorted(list(set_seg - set_full))
    only_in_full     = sorted(list(set_full - set_seg))
    in_both          = sorted(list(set_full & set_seg))

    print("부분에서만 등장하는 키워드 ({})".format(len(only_in_segments)))
    print("  : " + (", ".join(only_in_segments) if only_in_segments else "-"))
    print("\n전체에서만 등장하는 키워드 ({})".format(len(only_in_full)))
    print("  : " + (", ".join(only_in_full) if only_in_full else "-"))
    print("\n공통 키워드 ({})".format(len(in_both)))
    if in_both:
        print("  : ", ", ".join(in_both[:10]))

    def score(df, key, col):
        row = df[df["keyword"] == key]
        return float(row[col].values[0]) if len(row) else None

    rows=[]
    for k in in_both:
        rows.append({
            "keyword": k,
            "full_score":  score(df_full,  k, "similarity_full"),
            "left_score":  score(df_left,  k, "similarity_left"),
            "right_score": score(df_right, k, "similarity_right"),
        })
    df_overlap = pd.DataFrame(rows)
    if not df_overlap.empty:
        df_overlap["segment_best"] = df_overlap[["left_score","right_score"]].max(axis=1, skipna=True)
        df_overlap = df_overlap.sort_values(["full_score","segment_best"], ascending=False, na_position="last").reset_index(drop=True)

    print("\n전체 vs 부분 공통 키워드 점수 비교표")
    if df_overlap.empty:
        print("  - 공통 키워드가 없습니다.")
    else:
        display(df_overlap)

    stronger = []
    for _, r in df_overlap.iterrows():
        fs, sb = r.get("full_score"), r.get("segment_best")
        if (fs is not None) and (sb is not None) and not (math.isnan(fs) or math.isnan(sb)):
            if sb >= fs - 1e-6:
                stronger.append(r["keyword"])

    return only_in_segments, only_in_full, df_overlap

_ = summarize_keybert_results(df_full, df_left, df_right)
