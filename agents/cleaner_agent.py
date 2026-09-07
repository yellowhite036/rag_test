"""
CleanerAgent – 確定性格式清理器（不呼叫 LLM）。

職責：
- 把 CoderAgent 的原始輸出標準化成正確的 FILE_START/FILE_END 區塊
- 不依賴模型「這次有沒有乖乖照格式輸出」
- 清理失敗時回傳 ([], False)，呼叫方應重試 CoderAgent

擴充自 assistant.py 的 _clean_model_output()，
加入：換行符正規化、BOM 清除、多餘空行壓縮、
      多層 code fence 剝除、分區塊逐段修補。
"""
from __future__ import annotations

import re
from typing import NamedTuple

# 標準格式：##### FILE_START: path/to/file.py #####\n...\n##### FILE_END #####
_STANDARD_PATTERN = re.compile(
    r"##### FILE_START: (.+?) #####\n(.*?)\n##### FILE_END #####",
    re.DOTALL,
)

# 最外層 code fence（```python ... ``` 或 ``` ... ```）
_OUTER_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)

# 寬鬆 FILE_START/FILE_END（沒有 #####，或 ###### 數量不對）
_LOOSE_PATTERN = re.compile(
    r"#{2,6}\s*FILE_START[:\s]*([^\n]*)\n(.*?)\n#{2,6}\s*FILE_END[^\n]*",
    re.DOTALL,
)

# 更寬鬆：裸文字 FILE_START
_BARE_PATTERN = re.compile(
    r"FILE_START[:\s]+([^\n]+)\n(.*?)\nFILE_END",
    re.DOTALL,
)


class CleanResult(NamedTuple):
    matches: list[tuple[str, str]]  # [(rel_path, content), ...]
    ok: bool                        # 是否成功解析出至少一個區塊
    warnings: list[str]             # 非致命的清理提示


def _strip_outer_fence(text: str) -> str:
    """剝除把整段輸出包起來的最外層 code fence。"""
    stripped = text.strip()
    m = _OUTER_FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _normalize_newlines(text: str) -> str:
    """統一換行符，去除 BOM。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff")
    return text


def _compress_blank_lines(text: str) -> str:
    """把連續 3 行以上的空行壓縮成 2 行（保留段落感，避免過度壓縮）。"""
    return re.sub(r"\n{3,}", "\n\n", text)


def _rebuild_from_loose(text: str) -> tuple[str | None, str]:
    """
    嘗試用寬鬆 pattern 找到區塊，重建成標準格式。
    回傳 (重建後文字 | None, 使用的 pattern 名稱)。
    """
    # 先試 ##### 數量不對的情況
    loose_matches = _LOOSE_PATTERN.findall(text)
    if loose_matches:
        blocks = [
            f"##### FILE_START: {rel.strip()} #####\n{content.strip()}\n##### FILE_END #####"
            for rel, content in loose_matches
        ]
        return "\n\n".join(blocks), "loose_hash"

    # 再試裸 FILE_START
    bare_matches = _BARE_PATTERN.findall(text)
    if bare_matches:
        blocks = [
            f"##### FILE_START: {rel.strip()} #####\n{content.strip()}\n##### FILE_END #####"
            for rel, content in bare_matches
        ]
        return "\n\n".join(blocks), "bare"

    return None, ""


def _strip_per_block_fences(text: str) -> str:
    """
    有時模型會在每個 FILE_START 區塊的內容外面再包一層 code fence：
    ##### FILE_START: foo.py #####
    ```python
    def foo(): ...
    ```
    ##### FILE_END #####
    → 把每個區塊裡的內容部分的 code fence 剝掉。
    """
    def _strip_block_fence(m: re.Match) -> str:
        rel = m.group(1)
        content = m.group(2)
        inner = _OUTER_FENCE_RE.match(content.strip())
        if inner:
            content = inner.group(1)
        return f"##### FILE_START: {rel} #####\n{content.strip()}\n##### FILE_END #####"

    return _STANDARD_PATTERN.sub(_strip_block_fence, text)


class CleanerAgent:
    """
    確定性格式清理 Agent。不呼叫 LLM。
    """

    def run(self, raw_output: str) -> CleanResult:
        """
        清理 CoderAgent 的原始輸出，回傳 CleanResult。
        """
        warnings: list[str] = []
        text = _normalize_newlines(raw_output)
        text = _strip_outer_fence(text)
        text = _compress_blank_lines(text)

        # 先嘗試直接解析
        if _STANDARD_PATTERN.search(text):
            # 可能區塊內容有 code fence，再剝一次
            text = _strip_per_block_fences(text)
            matches = _STANDARD_PATTERN.findall(text)
            if matches:
                return CleanResult(
                    matches=[(rel.strip(), content) for rel, content in matches],
                    ok=True,
                    warnings=warnings,
                )

        # 嘗試寬鬆修補
        rebuilt, pattern_used = _rebuild_from_loose(text)
        if rebuilt:
            warnings.append(f"格式不標準（已用 {pattern_used} 模式修補）")
            matches = _STANDARD_PATTERN.findall(rebuilt)
            if matches:
                return CleanResult(
                    matches=[(rel.strip(), content) for rel, content in matches],
                    ok=True,
                    warnings=warnings,
                )

        # 完全解析失敗
        return CleanResult(matches=[], ok=False, warnings=["無法解析任何 FILE_START/FILE_END 區塊"])
