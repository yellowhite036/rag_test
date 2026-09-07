"""
CoderAgent – 負責呼叫 LLM 產出程式碼修改。

設計原則：
- Prompt 只給「方向」，不在 prompt 裡強制格式規則
  （格式強制交給 CleanerAgent，這樣即使模型違規也能修補）
- 重試時只傳 spec + 原始程式碼 + QA 失敗原因，context 大小固定不膨脹
- 每次重試都是乾淨的獨立呼叫（無 chat history 累積）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama


@dataclass
class CoderResult:
    raw_output: str
    attempt: int         # 第幾次嘗試（1-based）
    ok: bool = True      # 是否成功收到非空回覆


class CoderAgent:
    """
    編碼 Agent。

    Parameters
    ----------
    model:      Ollama 模型名稱
    num_ctx:    模型 context window 大小（tokens）
    temperature: 生成溫度（修改任務建議 0）
    """

    # Prompt 模板：只給方向，格式範例最小化（不放強制規則）
    _SYSTEM_PROMPT = (
        "你是一個程式碼修改助手，只輸出修改後的完整檔案內容，"
        "每個檔案用 ##### FILE_START: 檔名 ##### 開頭、##### FILE_END ##### 結尾。"
    )

    _USER_TEMPLATE = """\
【需求】
{requirement}

【原始程式碼】
{files_text}

請輸出所有檔案修改後的完整內容（包含未修改的函式），格式：
##### FILE_START: 檔名 #####
...完整內容...
##### FILE_END #####
"""

    _RETRY_SUFFIX = """\

【上一次 QA 失敗，請修正以下問題後重新輸出完整內容】
{qa_problems}
"""

    def __init__(
        self,
        model: str,
        num_ctx: int = 16384,
        temperature: float = 0.0,
    ):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature

    def run(
        self,
        spec: dict[str, Any],
        files_text: str,
        qa_problems: list[str] | None = None,
        attempt: int = 1,
    ) -> CoderResult:
        """
        呼叫 LLM 產出修改。

        Parameters
        ----------
        spec:         SpecAgent 輸出的規格 dict
        files_text:   所有目標檔案的 FILE_START/FILE_END 打包文字
        qa_problems:  上一次 QA 失敗的問題清單（第一次呼叫傳 None 或 []）
        attempt:      當前是第幾次嘗試（用於 log）
        """
        requirement = spec.get("requirement", "")

        # 加入 constraints（如果有的話）
        constraints = spec.get("constraints", [])
        if constraints:
            constraint_block = "\n".join(f"- {c}" for c in constraints)
            requirement = f"{requirement}\n\n【限制條件（不可修改）】\n{constraint_block}"

        user_content = self._USER_TEMPLATE.format(
            requirement=requirement,
            files_text=files_text,
        )

        # 重試時附上 QA 失敗原因，不附上上次的輸出（避免 context 膨脹）
        if qa_problems:
            problems_text = "\n".join(f"- {p}" for p in qa_problems)
            user_content += self._RETRY_SUFFIX.format(qa_problems=problems_text)

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                options={
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                },
            )
            raw = resp["message"]["content"]
            return CoderResult(raw_output=raw, attempt=attempt, ok=bool(raw.strip()))
        except Exception as e:
            return CoderResult(raw_output=f"[CoderAgent 呼叫失敗: {e}]", attempt=attempt, ok=False)
