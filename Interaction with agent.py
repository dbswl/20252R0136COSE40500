
from google.colab import drive
drive.mount('/content/drive')

! pip install langchain langchain-openai langchain-core langgraph pydantic
!apt-get -y install graphviz graphviz-dev
!pip install pygraphviz

import os

os.environ['OPENAI_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NjQ4NDczMjMsIm5iZiI6MTc2NDg0NzMyMywiZXhwIjoxNzY1NTUxNTk5LCJrZXlfaWQiOiI3OWIwY2U4Ny1lYzhkLTQ4MGMtYTZlYy1hMzE1NDdiMWNjYzgifQ.2t4RJv1XWTuHkEzmSx--T_JEn_JZ52Sjp6-5qvoeMF0'
os.environ['OPENAI_BASE_URL'] = 'https://mlapi.run/daef5150-72ef-48ff-8861-df80052ea7ac/v1'
# Key expiration date 2025/12/12 Fri.

import os
import random
import base64
import json
import mimetypes
from typing_extensions import TypedDict
from typing import Dict, Tuple

from pydantic import BaseModel, Field, conint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image, display



class InitialStats(BaseModel):
    intelligence: conint(ge=50, le=70)
    stamina: conint(ge=50, le=70)
    mentality: conint(ge=50, le=70)
    wealth: conint(ge=50, le=70)


class TargetStats(BaseModel):
    intelligence: conint(ge=70, le=90)
    stamina: conint(ge=70, le=90)
    mentality: conint(ge=70, le=90)
    wealth: conint(ge=70, le=90)


class Affinity(BaseModel):
    intelligence: conint(ge=0, le=100)
    stamina: conint(ge=0, le=100)
    mentality: conint(ge=0, le=100)
    wealth: conint(ge=0, le=100)


class Delight(BaseModel):
    intelligence: conint(ge=0, le=100)
    stamina: conint(ge=0, le=100)
    mentality: conint(ge=0, le=100)
    wealth: conint(ge=0, le=100)

class GameState(TypedDict):
    name: str
    intro: str
    goal: str
    current: Dict[str, int]
    target: Dict[str, int]
    affinity: Dict[str, int]
    turn: int
    cycle: int
    actions_this_cycle: int
    last_action: str | None        # 🔹 마지막 액션 이름
    repeat_count: int              # 🔹 같은 액션 연속 사용 횟수


LABELS = {
    "intelligence": "Intelligence",
    "stamina": "Stamina",
    "mentality": "Mentality",
    "wealth": "Wealth",
}

ACTIONS = {
    "study": {"plus": "intelligence", "minus": "stamina"},
    "workout": {"plus": "stamina", "minus": "intelligence"},
    "hobby": {"plus": "mentality", "minus": "wealth"},
    "part-time": {"plus": "wealth", "minus": "mentality"},

    # travel-plan: +Mentality, -Wealth, small +Intelligence
    "travel-plan": {"plus": "mentality", "minus": "wealth"},

    # networking: +Wealth, small +Mentality, -Stamina
    "networking": {"plus": "wealth", "minus": "stamina"},
}

VALID_ACTIONS = list(ACTIONS.keys())

MIN_TRAVEL_WEALTH = 40

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def print_stats(current: Dict[str, int], target: Dict[str, int]) -> None:
    bar = "-" * 44
    print(bar)
    print("Current stats / Target stats")
    for k in ["intelligence", "stamina", "mentality", "wealth"]:
        print(f"  {LABELS[k]:<11}: {current[k]:>3}  /  {target[k]:>3}")
    print(bar)


def print_affinity(aff: Dict[str, int]) -> None:
    print("Affinities (0–100, higher = likes more)")
    for k in ["intelligence", "stamina", "mentality", "wealth"]:
        print(f"  {LABELS[k]:<11}: {aff[k]:>3}")
    print("-" * 44)


def path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def show_image_debug(image_path: str) -> None:
    print("=== Gift image path ===")
    print(f"Image path: {image_path}")

    try:
        from PIL import Image
        try:
            from IPython.display import display
            img = Image.open(image_path)
            display(img)
        except Exception:
            img = Image.open(image_path)
            img.show()
    except Exception as e:
        print(f"[WARN] Failed to display image: {e}")


def parse_json_stripping_fences(text: str) -> dict:
    """
    Robustly extract JSON from a string that may be wrapped in ``` or ```json fences.
    We just grab the substring between the first '{' and the last '}'.
    """
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in: {text!r}")
    json_str = s[start : end + 1]
    return json.loads(json_str)



def make_llm(model_name: str | None = None) -> ChatOpenAI:
    selected = model_name or os.getenv("OPENAI_MODEL", "openai/gpt-5-nano")
    return ChatOpenAI(model=selected)


def load_gift_text_from_path(gift_text_file_path: str | None) -> str | None:
    if not gift_text_file_path:
        return None

    if not os.path.exists(gift_text_file_path):
        print(f"[WARN] gift_text_file_path does not exist: {gift_text_file_path}")
        return None

    ext = os.path.splitext(gift_text_file_path)[1].lower()
    if ext != ".json":
        print(f"[WARN] Only .json is supported for gift_text_file_path (got: {ext})")
        return None

    try:
        with open(gift_text_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_parts = []

        if isinstance(data, dict):
            title = data.get("title")
            summary = data.get("summary")
            metadata = data.get("metadata")

            if title:
                text_parts.append(f"Title: {title}")
            if summary:
                text_parts.append(f"Summary: {summary}")
            if metadata and isinstance(metadata, dict):
                text_parts.append("Metadata:")
                for k, v in metadata.items():
                    text_parts.append(f"- {k}: {v}")

            if not text_parts:
                text_parts.append(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            text_parts.append(str(data))

        gift_text = "\n".join(text_parts)

        print("=== Gift text loaded from JSON path ===")
        print(f"Path: {gift_text_file_path}")
        print("----- Content start -----")
        print(gift_text)
        print("----- Content end -----")

        return gift_text

    except Exception as e:
        print(f"[ERROR] Failed to load gift text from {gift_text_file_path}: {e}")
        return None

def build_initial_chain(llm: ChatOpenAI):
    parser = PydanticOutputParser(pydantic_object=InitialStats)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Allocate initial stats based on the player's self-introduction. "
             "Return integer that is greater than 50 and less than 70 for Intelligence, Stamina, Mentality, Wealth. "
             "Respond ONLY in the required format."),
            ("user",
             "Self-introduction:\n{intro}\n\n"
             "Interpretation guide:\n"
             "- Intelligence: academics/learning/analytical interests\n"
             "- Stamina: sports/physical activity/sleep & routine\n"
             "- Mentality: hobbies/creativity/social-emotional balance\n"
             "- Wealth: part-time work/investing/business/spending habits\n\n"
             "{format_instructions}"),
        ]
    )
    chain = prompt | llm | parser
    return chain, parser


def build_target_chain(llm: ChatOpenAI):
    parser = PydanticOutputParser(pydantic_object=TargetStats)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Set target stats from the player's 'vacation goal'. "
             "Return integers (70–90) for Intelligence, Stamina, Mentality, Wealth. "
             "Ambitious → 75–90; conservative → 70–78. ONLY required format."),
            ("user", "Vacation goal:\n{goal}\n\n{format_instructions}"),
        ]
    )
    chain = prompt | llm | parser
    return chain, parser


def analyze_intro_and_goal(intro: str, goal: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    llm = make_llm()
    init_chain, init_parser = build_initial_chain(llm)
    tgt_chain, tgt_parser = build_target_chain(llm)

    init_obj: InitialStats = init_chain.invoke(
        {"intro": intro, "format_instructions": init_parser.get_format_instructions()}
    )
    tgt_obj: TargetStats = tgt_chain.invoke(
        {"goal": goal, "format_instructions": tgt_parser.get_format_instructions()}
    )

    initial = {
        "intelligence": int(init_obj.intelligence),
        "stamina": int(init_obj.stamina),
        "mentality": int(init_obj.mentality),
        "wealth": int(init_obj.wealth),
    }
    target = {
        "intelligence": int(tgt_obj.intelligence),
        "stamina": int(tgt_obj.stamina),
        "mentality": int(tgt_obj.mentality),
        "wealth": int(tgt_obj.wealth),
    }
    for k in initial: initial[k] = clamp(initial[k], 50, 70)
    for k in target:  target[k]  = clamp(target[k], 70, 90)
    return initial, target


def analyze_affinity(intro_text: str) -> Dict[str, int]:
    """
    Player's baseline liking (0..100) for each domain, derived from self-intro.
    """
    llm = make_llm()
    system = SystemMessage(
        content=("Infer how much the player LIKES each stat domain from their self-introduction. "
                 "Return integers 0–100 for intelligence, stamina, mentality, wealth. "
                 "0 = dislikes/avoidant, 50 = neutral, 100 = loves it. "
                 "Strict JSON only.")
    )
    human = HumanMessage(content=[
        {"type": "text", "text": f"Self-introduction:\n{intro_text}\n"},
        {"type": "text", "text": (
            "JSON schema:\n"
            '{ "intelligence": 0-100, "stamina": 0-100, "mentality": 0-100, "wealth": 0-100 }\n'
            "Return ONLY the JSON."
        )}
    ])
    resp = llm.invoke([system, human])
    try:
        data = parse_json_stripping_fences(resp.content)
        aff = Affinity.model_validate(data)
        return aff.model_dump()
    except Exception:
        return {"intelligence": 50, "stamina": 50, "mentality": 50, "wealth": 50}


def analyze_gift_delight(
    intro_text: str,
    player_affinity: Dict[str, int],
    gift_text_file_path: str | None,
    gift_image_path: str | None,
) -> Dict[str, int]:
    """
    Works for:
      - JSON-based gifts (book metadata + summary) + image
      - image-only gifts
    """
    if not gift_text_file_path and not gift_image_path:
        raise ValueError(
            "At least one of gift_text_file_path or gift_image_path must be provided. "
            "Text-only gifts are not supported."
        )

    llm = make_llm()
    system = SystemMessage(
        content=(
            "You are a gift matcher. Given the player's self-introduction and baseline affinities, "
            "and the gift (text and/or image), estimate HOW MUCH the player will LIKE this gift "
            "in each stat domain. Output STRICT JSON with integers 0–100 for keys: "
            "intelligence, stamina, mentality, wealth. "
            "Consider both topical relevance and motivational pull. No commentary."
        )
    )

    gift_text: str | None = load_gift_text_from_path(gift_text_file_path)

    contents: list[dict] = []
    contents.append({"type": "text", "text": f"Player self-introduction:\n{intro_text}\n"})
    contents.append(
        {
            "type": "text",
            "text": f"Baseline affinities (0..100): {json.dumps(player_affinity)}",
        }
    )

    if gift_text:
        contents.append({"type": "text", "text": f"Gift (text from JSON):\n{gift_text}\n"})

    if gift_image_path:
        show_image_debug(gift_image_path)
        try:
            data_url = path_to_data_url(gift_image_path)
            contents.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )
        except Exception as e:
            print(f"(Image load failed: {e})")
            contents.append(
                {"type": "text", "text": f"(Image load failed: {e})"}
            )

    contents.append(
        {
            "type": "text",
            "text": (
                'Return ONLY JSON like '
                '{"intelligence":0-100,"stamina":0-100,"mentality":0-100,"wealth":0-100}.'
            ),
        }
    )

    resp = llm.invoke([system, HumanMessage(content=contents)])

    raw = resp.content
    if isinstance(raw, list):
        text_parts: list[str] = []
        for part in raw:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            else:
                text_parts.append(str(part))
        raw_text = "\n".join(text_parts)
    else:
        raw_text = str(raw)

    try:
        data = parse_json_stripping_fences(raw_text)
        out: Dict[str, int] = {}
        for k in ["intelligence", "stamina", "mentality", "wealth"]:
            v = data.get(k, 50)
            try:
                v_int = int(v)
            except Exception:
                v_int = 50
            out[k] = clamp(v_int, 0, 100)
        return out
    except Exception:
        print("Neutral fallback if anything goes wrong in parsing")
        return {
            "intelligence": 50,
            "stamina": 50,
            "mentality": 50,
            "wealth": 50,
        }


def delight_to_bonus(d: int) -> int:
    """Map 0..100 → 1..15 deterministically."""
    return clamp(round(clamp(d, 0, 100) * 15 / 100), 1, 15)

def apply_action(state: Dict, action_name: str):
    stats = state["current"]
    target = state["target"]
    affinity = state.get("affinity", {})

    conf = ACTIONS[action_name]
    plus_key = conf["plus"]
    minus_key = conf["minus"]

    # -------------------------------
    # [1] 목표와의 거리 기반 보상/패널티 조절
    # -------------------------------
    # plus stat이 target과 얼마나 떨어져 있는지에 따라 보상 크기 조절
    gap_plus = max(0, target[plus_key] - stats[plus_key])  # 목표보다 낮을수록 gap 큼
    if gap_plus >= 20:
        base_reward = 10           # 목표에서 멀면 크게 보상
    elif gap_plus >= 10:
        base_reward = 8
    else:
        base_reward = 5            # 이미 목표에 가깝다면 보상 작게

    reward = base_reward

    # minus stat이 target보다 얼마나 높은지에 따라 패널티 크기 조절
    gap_minus = stats[minus_key] - target[minus_key]   # 현재 - 목표
    if gap_minus >= 15:
        base_penalty = 8       # 목표보다 현재 stat 훨씬 높다면 패널티 크게
    elif gap_minus >= 5:
        base_penalty = 6
    else:
        base_penalty = 4       # 목표보다 낮거나 비슷하다면 패널티 완화

    penalty = base_penalty


    # -----------------------------------------
    # [2] 같은 액션 반복 시 보상 감소
    # -----------------------------------------
    # state에 키가 없을 가능성도 있으니 안전하게 초기화
    last_action = state.get("last_action")
    repeat_count = state.get("repeat_count", 0)

    if last_action == action_name:
        repeat_count += 1
    else:
        repeat_count = 1

    state["last_action"] = action_name
    state["repeat_count"] = repeat_count

    # 2번 연속이면 보상 -2, 3번 이상이면 추가로 -2
    if repeat_count >= 2:
        reward -= 2
    if repeat_count >= 3:
        reward -= 2
    reward = clamp(reward, 3, 12)

    # -----------------------------------------
    # [3] Stamina가 너무 낮을 때 workout 보상 강화
    # -----------------------------------------
    if action_name == "workout" and stats["stamina"] < target["stamina"] - 15:
        reward += 2            # 회복을 유도
        reward = clamp(reward, 3, 12)

    # -----------------------------------------
    # [4] Affinity 기반 보상 조정
    # -----------------------------------------
    # 자기소개에 적은 성향을 바탕으로 조절
    # stamina를 좋아하면 workout 보상이 높아지는 식
    aff_value = affinity.get(plus_key, 50)  # 0~100, 50 = 중립
    aff_factor = 1 + (aff_value - 50) / 200  # 50 → 1.0, 100 → 1.25, 0 → 0.75 정도
    reward = int(round(reward * aff_factor))
    reward = clamp(reward, 3, 15)

    # -----------------------------------------
    # [5] 기본 plus/minus 스탯 적용
    # -----------------------------------------
    stats[plus_key] = max(0, stats[plus_key] + reward)
    stats[minus_key] = max(0, stats[minus_key] - penalty)

    extra_info = ""

    # -----------------------------------------
    # [6] Optional: 새 액션 보너스 처리
    #     - travel-plan: 작은 Intelligence 보너스
    #     - networking: 작은 Mentality 보너스
    # -----------------------------------------
    if action_name == "travel-plan":
        small_bonus = max(1, reward // 2)  # 보상 절반 정도를 Intelligence에
        stats["intelligence"] = max(0, stats["intelligence"] + small_bonus)
        extra_info = f", {LABELS['intelligence']} +{small_bonus}"

    if action_name == "networking":
        small_bonus = max(1, reward // 2) #보상 절반 정도를 mentality에
        stats["mentality"] = max(0, stats["mentality"] + small_bonus)
        extra_info = f", {LABELS['mentality']} +{small_bonus}"

    return reward, penalty, plus_key, minus_key, extra_info


def normalize_action(text: str) -> str:
    t = text.strip().lower()

    if t in {"part time", "parttime", "part_time"}:
        t = "part-time"
    if t in {"travel plan", "travelplan", "travel_plan"}:
        t = "travel-plan"

    return t


def run_gift_event_content_based(state: Dict) -> Dict[str, int]:
    """
    Ask for gift input (text or path). If non-empty, compute per-stat delight (0..100) based on content
    and deterministically map to bonuses 1..15 each.
    """
    print("\n🎁 Gift Event!")
    print("Enter either:")
    print(" - A short text describing the gift (e.g., 'a calculus textbook'), or")
    print(" - A local image path (e.g., './gift.jpg').")
    gift_in = input("Gift input: ").strip()

    if not gift_in:
        print("No input detected. No bonuses granted.")
        return state["current"]

    gift_text, gift_image = None, None
    if 'png' in gift_in.lower() or 'jpg' in gift_in.lower():
        gift_image = gift_in
    else:
        gift_text = gift_in

    delight = analyze_gift_delight(state["intro"], state["affinity"], gift_text, gift_image)
    bonuses = {k: delight_to_bonus(v) for k, v in delight.items()}

    for k, v in bonuses.items():
        state["current"][k] = max(0, state["current"][k] + v)

    bonus_list = ", ".join([f"{LABELS[k]} +{v}" for k, v in bonuses.items()])
    print(f"Gift delight (0–100): {delight}")
    print("Bonuses applied:", bonus_list)
    return state["current"]

def initial_state() -> Dict:
    return {
        "name": "",
        "intro": "",
        "goal": "",
        "current": {},
        "target": {},
        "affinity": {},
        "turn": 0,               # 0..11
        "cycle": 0,              # 0..3
        "actions_this_cycle": 0, # 0..3
        "last_action": None,     # 🔹 초기에는 없음
        "repeat_count": 0,       # 🔹 0회
    }



def node_ask_inputs(state: Dict) -> Dict:
    print("===== College Life (LangGraph + Content-Based Gifts) =====")
    state["name"] = input("Character name: ").strip()
    state["intro"] = input("Self-introduction (brief, in English): ").strip()
    state["goal"] = input("This vacation's goal (brief, in English): ").strip()
    return state


def node_setup_stats_and_affinity(state: Dict) -> Dict:
    current, target = analyze_intro_and_goal(state["intro"], state["goal"])
    state["current"], state["target"] = current, target
    state["affinity"] = analyze_affinity(state["intro"])

    print(f"\n{state['name']}'s character is ready!")
    print_stats(state["current"], state["target"])
    print_affinity(state["affinity"])
    return state


def node_do_action(state: Dict) -> Dict:
    if state["actions_this_cycle"] == 0:
        print(f"\n=== Cycle {state['cycle'] + 1}/3 ===")

    print(f"\n[Turn {state['turn'] + 1}/12] Action {state['actions_this_cycle'] + 1}/3")

    # --------------------------------------------------------
    # 🔍 Hint: Stamina 낮으면 workout 보상 강화 안내
    # --------------------------------------------------------
    if state["current"]["stamina"] < state["target"]["stamina"] - 15:
        print("⚡ Hint: Your Stamina is quite low.")
        print("   → Choosing 'workout' will grant EXTRA REWARD this turn!\n")


    print("Available actions: [study, workout, hobby, part-time, travel-plan, networking]")

    # --------------------------------------------------------
    # 사용자 입력 루프 (조건 위반 시 다시 선택)
    # --------------------------------------------------------
    while True:
        action = normalize_action(input("Choose your action: "))

        if action not in VALID_ACTIONS:
            print("Please type a valid action name.")
            continue

        # Conditional rule: travel-plan은 wealth가 충분할 때만 가능
        if action == "travel-plan":
            wealth = state["current"]["wealth"]
            if wealth < MIN_TRAVEL_WEALTH:
                print(
                    f"❌ Not enough Wealth for 'travel-plan'. "
                    f"(need ≥ {MIN_TRAVEL_WEALTH}, current = {wealth})"
                )
                continue  # 다시 선택하게 함

        # 조건 통과
        break

    # --------------------------------------------------------
    # 보상 적용
    # --------------------------------------------------------
    reward, penalty, plus_key, minus_key, extra_info = apply_action(state, action)

    print(
        f"▶ {action.capitalize()}: "
        f"{LABELS[plus_key]} +{reward}, {LABELS[minus_key]} -{penalty}{extra_info}"
    )
    print_stats(state["current"], state["target"])

    state["turn"] += 1
    state["actions_this_cycle"] += 1
    return state





def node_gift(state: Dict) -> Dict:
    print(f"\n[Turn {state['turn'] + 1}/12] Gift Event")
    state["current"] = run_gift_event_content_based(state)
    print_stats(state["current"], state["target"])

    state["turn"] += 1
    state["cycle"] += 1
    state["actions_this_cycle"] = 0
    return state


def decide_after_action(state: Dict) -> str:
    return "do_action" if state["actions_this_cycle"] < 3 else "gift"


def decide_after_gift(state: Dict) -> str:
    return "do_action" if state["cycle"] < 3 else "end"


def node_end(state: Dict) -> Dict:
    print("\n===== Game Over =====")
    print_stats(state["current"], state["target"])
    success = all(state["current"][k] >= state["target"][k] for k in state["current"])
    if success:
        print("🎉 You reached ALL targets!")
    else:
        print("😢 Some targets were not reached.")
    return state

def build_graph():
    g = StateGraph(GameState)
    g.add_node("ask_inputs", node_ask_inputs)
    g.add_node("setup_stats_and_affinity", node_setup_stats_and_affinity)
    g.add_node("do_action", node_do_action)
    g.add_node("gift", node_gift)
    g.add_node("end", node_end)

    g.set_entry_point("ask_inputs")
    g.add_edge("ask_inputs", "setup_stats_and_affinity")
    g.add_edge("setup_stats_and_affinity", "do_action")

    g.add_conditional_edges(
        "do_action",
        decide_after_action,
        {"do_action": "do_action", "gift": "gift"},
    )
    g.add_conditional_edges(
        "gift",
        decide_after_gift,
        {"do_action": "do_action", "end": "end"},
    )
    g.add_edge("end", END)

    return g.compile()

from IPython.display import Image, display

app = build_graph()
graph = app.get_graph()

png = graph.draw_png()
display(Image(png))

if __name__ == "__main__":
    random.seed()
    app = build_graph()
    app.invoke(initial_state())
