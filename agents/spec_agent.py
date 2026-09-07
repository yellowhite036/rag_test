"""
SpecAgent – 負責把使用者的原始指令解析成結構化規格。

策略：
- 預設使用確定性規則解析（速度快、上下文短）
- 偵測到複雜結構（多段中文標籤）時才升級呼叫 LLM
- 永遠回傳同一個 dict schema，讓後續 Agent 不需要判斷來源
"""
from __future__ import annotations

import json
import re
from typing import Any

import ollama


# 解析【驗收條件】/ 【需求】等中文結構標籤
_ACCEPTANCE_RE = re.compile(r"【驗收條件】(.*)", re.DOTALL)
_CONSTRAINT_RE = re.compile(r"【限制條件】(.*?)(?=【|$)", re.DOTALL)
_REQUIREMENT_TAG_RE = re.compile(r"^【需求】\s*", re.MULTILINE)

# 偵測「是否有複雜多段標籤結構」的啟發式判斷
_COMPLEX_STRUCTURE_RE = re.compile(r"【[^】]+】", re.MULTILINE)


def _rule_parse(raw: str) -> dict[str, Any]:
    """
    確定性規則解析（不呼叫 LLM）。
    抓取【需求】【驗收條件】【限制條件】三個標籤；
    沒有標籤的純文字直接當作 requirement。
    """
    spec: dict[str, Any] = {
        "requirement": raw.strip(),
        "acceptance": [],
        "constraints": [],
        "target_files": [],
    }

    # 抓驗收條件
    acc_match = _ACCEPTANCE_RE.search(raw)
    if acc_match:
        spec["requirement"] = raw[: acc_match.start()].strip()
        spec["requirement"] = _REQUIREMENT_TAG_RE.sub("", spec["requirement"]).strip()
        spec["acceptance"] = [
            line.strip().lstrip("-").strip()
            for line in acc_match.group(1).splitlines()
            if line.strip().startswith("-")
        ]

    # 抓限制條件
    con_match = _CONSTRAINT_RE.search(raw)
    if con_match:
        spec["constraints"] = [
            line.strip().lstrip("-").strip()
            for line in con_match.group(1).splitlines()
            if line.strip().startswith("-")
        ]
        # 若 requirement 裡混入了限制條件區段，清除
        spec["requirement"] = re.sub(r"【限制條件】.*", "", spec["requirement"], flags=re.DOTALL).strip()

    return spec


def _llm_parse(raw: str, model: str) -> dict[str, Any] | None:
    """
    用 LLM 將自由文字解析成結構化 JSON spec。
    失敗時回傳 None，讓呼叫方 fallback 到規則解析。
    """
    prompt = f"""你是需求分析師，請把以下使用者指令整理成 JSON 格式，不要加任何說明文字。

使用者指令：
{raw}

請輸出以下格式的 JSON（如果某個欄位沒有資訊就給空陣列或空字串）：
{{
  "requirement": "清楚的需求描述（一段話）",
  "acceptance": ["驗收條件1", "驗收條件2"],
  "constraints": ["不能動的函式或邏輯"],
  "target_files": ["推測可能需要修改的檔名，沒有就留空陣列"]
}}

只輸出 JSON，不要輸出其他文字。"""

    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 4096},
        )
        text = resp["message"]["content"].strip()
        # 剝除可能的 markdown fence
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        data = json.loads(text)
        # 正規化欄位
        return {
            "requirement": str(data.get("requirement", raw.strip())),
            "acceptance": list(data.get("acceptance", [])),
            "constraints": list(data.get("constraints", [])),
            "target_files": list(data.get("target_files", [])),
        }
    except Exception:
        return None


class SpecAgent:
    """
    規格解析 Agent。

    - 先用規則解析
    - 若規則解析後 acceptance 是空的、且指令字數超過 100 字，
      且 `use_llm=True`，則升級用 LLM 解析
    """

    def __init__(self, model: str, use_llm: bool = False):
        self.model = model
        self.use_llm = use_llm

    def run(self, raw_instruction: str) -> dict[str, Any]:
        """
        回傳結構化 spec dict：
        {
            "requirement": str,
            "acceptance": list[str],
            "constraints": list[str],
            "target_files": list[str],
            "_source": "rule" | "llm" | "llm_fallback"
        }
        """
        spec = _rule_parse(raw_instruction)

        # 判斷是否需要升級到 LLM
        should_upgrade = (
            self.use_llm
            and not spec["acceptance"]
            and len(raw_instruction) > 100
            # 沒有標準化標籤，純自由文字
            and not _COMPLEX_STRUCTURE_RE.search(raw_instruction)
        )

        if should_upgrade:
            llm_spec = _llm_parse(raw_instruction, self.model)
            if llm_spec:
                llm_spec["_source"] = "llm"
                return llm_spec
            # LLM 失敗，退化回規則解析結果
            spec["_source"] = "llm_fallback"
        else:
            spec["_source"] = "rule"

        return spec
