"""
pipeline.py – 組裝 4 個 Agent 的主流程，替換 assistant.py 的 auto_edit_with_local_model()。

流程：
    [SpecAgent] → [CoderAgent] → [CleanerAgent] → [QAAgent]
                      ↑_____(失敗時帶著問題清單重試)_______|

重試策略：
- CleanerAgent 格式解析失敗 → 重試 CoderAgent（不帶 QA 問題，只帶格式提示）
- QAAgent 失敗             → 重試 CoderAgent（帶具體 QA 問題清單）
- 每次重試 prompt context 大小固定（不累積上一次輸出）
"""
from __future__ import annotations

import sys
import difflib
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.spec_agent import SpecAgent
from agents.coder_agent import CoderAgent
from agents.cleaner_agent import CleanerAgent
from agents.qa_agent import QAAgent


# ── 常數（使用端可 import 後覆蓋） ────────────────────────────────────────────
DEFAULT_MAX_RETRIES = 3


# ── 輔助函式 ──────────────────────────────────────────────────────────────────

def _build_files_text(files: list[Path], files_dir: Path) -> tuple[str, set[str]]:
    """
    把 files/ 底下所有 .py 檔打包成 FILE_START/FILE_END 格式。
    回傳 (打包後文字, 已知相對路徑集合)。
    """
    blocks: list[str] = []
    known_files: set[str] = set()
    for f in files:
        rel = str(f.relative_to(files_dir))
        known_files.add(rel)
        content = f.read_text(encoding="utf-8", errors="replace")
        blocks.append(
            f"##### FILE_START: {rel} #####\n{content}\n##### FILE_END #####"
        )
    return "\n\n".join(blocks), known_files


def _show_diff(rel_path: str, new_content: str, files_dir: Path) -> None:
    """顯示單一檔案的 unified diff。"""
    target = files_dir / rel_path
    old_content = (
        target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
    )
    diff_lines = list(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"目前版本: {rel_path}",
            tofile=f"修改後版本: {rel_path}",
        )
    )
    if diff_lines:
        print(f"\n--- {rel_path} 的變更 ---")
        for line in diff_lines:
            print(line, end="" if line.endswith("\n") else "\n")
    else:
        print(f"\n--- {rel_path} 無變化 ---")


def _validate_filenames(matches: list[tuple[str, str]], known_files: set[str]) -> list[str]:
    """確定性驗證：檔名是否合理（存在或至少是 .py）。"""
    problems: list[str] = []
    for rel_path, _ in matches:
        rel_path = rel_path.strip()
        if not rel_path:
            problems.append("出現空白檔名")
            continue
        if rel_path not in known_files and not rel_path.endswith(".py"):
            problems.append(f"可疑檔名（不存在且副檔名不是 .py）: {rel_path}")
        if ".." in rel_path:
            problems.append(f"可疑檔名（包含 .. 路徑跳脫）: {rel_path}")
    return problems


def _write_files(
    matches: list[tuple[str, str]],
    files_dir: Path,
    self_file: str = "",
) -> None:
    """備份 + 寫入。"""
    backup_dir = files_dir / f".backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup_dir.mkdir(exist_ok=True)
    for rel_path, content in matches:
        rel_path = rel_path.strip()
        if self_file and Path(rel_path).name == self_file:
            print(f"已跳過（保護工具自身）: {rel_path}")
            continue
        target = files_dir / rel_path
        if target.exists():
            bkp = backup_dir / rel_path
            bkp.parent.mkdir(parents=True, exist_ok=True)
            bkp.write_text(
                target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"已更新: {rel_path}")
    print(f"備份於: {backup_dir}")


# ── 主 Pipeline ───────────────────────────────────────────────────────────────

class EditPipeline:
    """
    4-Agent 編輯 Pipeline。

    Parameters
    ----------
    llm_model:   Ollama 模型名稱
    files_dir:   目標檔案資料夾 (Path)
    py_files:    要送給 CoderAgent 的 .py 檔案清單 (list[Path])
    self_file:   assistant.py 本身的檔名（受保護，不會被寫入）
    max_retries: CoderAgent 最大重試次數
    use_llm_spec: SpecAgent 是否允許升級到 LLM 解析（預設 False）
    """

    def __init__(
        self,
        coder_model: str,
        files_dir: Path,
        py_files: list[Path],
        spec_model: str | None = None,
        self_file: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        use_llm_spec: bool = False,
        num_ctx: int = 16384,
    ):
        self.coder_model = coder_model
        self.spec_model = spec_model or coder_model
        self.files_dir = files_dir
        self.py_files = py_files
        self.self_file = self_file
        self.max_retries = max_retries

        # 初始化 4 個 Agent
        self.spec_agent = SpecAgent(model=self.spec_model, use_llm=use_llm_spec)
        self.coder_agent = CoderAgent(model=self.coder_model, num_ctx=num_ctx)
        self.cleaner_agent = CleanerAgent()
        self.qa_agent = QAAgent(files_dir=files_dir, self_file=self_file)

    def run(self, raw_instruction: str, flush_stdin_fn=None) -> bool:
        """
        執行完整 pipeline。
        回傳 True 表示成功寫入，False 表示失敗或使用者取消。

        flush_stdin_fn: 可選的 stdin 清空函式（傳入 assistant._flush_stdin）
        """

        def _flush():
            if flush_stdin_fn:
                flush_stdin_fn()

        # ── Step 1: SpecAgent ─────────────────────────────────────────────────
        print("\n[Agent 1/4] SpecAgent 解析規格...")
        spec = self.spec_agent.run(raw_instruction)
        print(f"  來源: {spec.get('_source', '?')}")
        print(f"  需求: {spec['requirement'][:120]}{'...' if len(spec['requirement']) > 120 else ''}")
        if spec["acceptance"]:
            print("  驗收條件:")
            for c in spec["acceptance"]:
                print(f"    - {c}")
        if spec["constraints"]:
            print("  限制條件:")
            for c in spec["constraints"]:
                print(f"    - {c}")
        if not spec["acceptance"]:
            print("  ⚠️  沒有驗收條件，QA 只能做確定性檢查，無法核對需求是否達成")

        _flush()
        confirm = input("\n確認用這段規格執行嗎?(y/n): ").strip()
        if confirm.lower() != "y":
            print("已取消")
            return False

        # ── 打包程式碼 ────────────────────────────────────────────────────────
        files_text, known_files = _build_files_text(self.py_files, self.files_dir)

        # ── Step 2~4: CoderAgent → CleanerAgent → QAAgent（帶重試）─────────────
        matches: list[tuple[str, str]] = []
        qa_problems: list[str] = []
        fail_reason = "format"  # "format" | "qa" | "filename"

        for attempt in range(1, self.max_retries + 1):
            # Step 2: CoderAgent
            print(f"\n[Agent 2/4] CoderAgent 產出程式碼 (第 {attempt}/{self.max_retries} 次)...")
            coder_result = self.coder_agent.run(
                spec=spec,
                files_text=files_text,
                qa_problems=qa_problems if attempt > 1 else None,
                attempt=attempt,
            )

            # 儲存 debug 用
            Path("debug_last_prompt.txt").write_text(
                coder_result.raw_output, encoding="utf-8"
            )

            if not coder_result.ok:
                print(f"  ❌ CoderAgent 呼叫失敗: {coder_result.raw_output[:200]}")
                qa_problems = ["CoderAgent 呼叫失敗，請重試"]
                fail_reason = "format"
                continue

            # Step 3: CleanerAgent
            print("[Agent 3/4] CleanerAgent 清理格式...")
            clean_result = self.cleaner_agent.run(coder_result.raw_output)

            if clean_result.warnings:
                for w in clean_result.warnings:
                    print(f"  ⚠️  {w}")

            if not clean_result.ok:
                print("  ❌ 格式解析失敗，無法取得任何 FILE_START/FILE_END 區塊")
                qa_problems = ["輸出格式完全無法解析，請務必用 ##### FILE_START: 檔名 ##### 開頭"]
                fail_reason = "format"
                if attempt < self.max_retries:
                    print(f"  → 將重試 CoderAgent（第 {attempt + 1} 次）")
                continue

            # 檔名合理性驗證
            filename_problems = _validate_filenames(clean_result.matches, known_files)
            if filename_problems:
                print("  ❌ 檔名驗證失敗:")
                for p in filename_problems:
                    print(f"    - {p}")
                qa_problems = filename_problems
                fail_reason = "filename"
                if attempt < self.max_retries:
                    print(f"  → 將重試 CoderAgent（第 {attempt + 1} 次）")
                continue

            # Step 4: QAAgent
            print("[Agent 4/4] QAAgent 多層確定性檢查...")
            qa_result = self.qa_agent.run(clean_result.matches)

            # 顯示警告（L4 複雜度等）
            if qa_result.warnings:
                print("  ⚠️  QA 警告（不阻止寫入）:")
                for w in qa_result.warnings:
                    print(f"    {w}")

            if qa_result.passed:
                print("  ✅ 所有 QA 層通過")
                matches = clean_result.matches
                break  # 成功，跳出重試迴圈
            else:
                print("  ❌ QA 未通過:")
                for e in qa_result.errors:
                    print(f"    - {e}")
                qa_problems = qa_result.errors
                fail_reason = "qa"
                if attempt < self.max_retries:
                    print(f"  → 將帶著 QA 失敗原因重試 CoderAgent（第 {attempt + 1} 次）")

        else:
            # 重試次數耗盡
            print(f"\n❌ 已重試 {self.max_retries} 次仍無法通過，放棄本次修改。")
            print(f"   最後失敗原因: {fail_reason}")
            if qa_problems:
                print("   最後問題清單:")
                for p in qa_problems:
                    print(f"   - {p}")
            return False

        # ── 顯示 Diff ──────────────────────────────────────────────────────────
        print("\n========== 變更內容 (diff) ==========")
        for rel_path, content in matches:
            _show_diff(rel_path.strip(), content, self.files_dir)
        print("=====================================")

        # ── 驗收條件提醒 ───────────────────────────────────────────────────────
        if spec["acceptance"]:
            print("\n📋 本次修改應滿足以下驗收條件（請對照上方 diff 自行確認）:")
            for c in spec["acceptance"]:
                print(f"  - {c}")

        # ── 使用者確認 ─────────────────────────────────────────────────────────
        _flush()
        confirm = input("\n確認套用嗎?(y/n): ").strip()
        if confirm.lower() != "y":
            print("已取消")
            return False

        # ── 寫入 ───────────────────────────────────────────────────────────────
        _write_files(matches, self.files_dir, self.self_file)
        return True


# ── 供 assistant.py 呼叫的頂層函式 ────────────────────────────────────────────

def run_edit_pipeline(
    raw_instruction: str,
    coder_model: str,
    files_dir: Path,
    py_files: list[Path],
    spec_model: str | None = None,
    self_file: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
    num_ctx: int = 16384,
    flush_stdin_fn=None,
) -> bool:
    """
    便捷入口：建立 EditPipeline 並執行。
    設計為可直接替換 assistant.py 裡的 auto_edit_with_local_model() 內部邏輯。
    """
    pipeline = EditPipeline(
        coder_model=coder_model,
        files_dir=files_dir,
        py_files=py_files,
        spec_model=spec_model,
        self_file=self_file,
        max_retries=max_retries,
        num_ctx=num_ctx,
    )
    return pipeline.run(raw_instruction, flush_stdin_fn=flush_stdin_fn)
