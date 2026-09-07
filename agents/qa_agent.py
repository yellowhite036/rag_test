"""
QAAgent – 確定性多層 Gate（不呼叫 LLM）。

Layer 1 (L1): AST 語法合法性
Layer 2 (L2): 函式是否無故消失
Layer 3 (L3): 必要 import 是否無故消失
Layer 4 (L4): Cyclomatic Complexity 是否明顯上升（警告，不擋寫入）
Layer 5 (L5): py_compile 字節碼編譯驗證（比 ast.parse 更嚴格）

L1/L2/L3/L5 不通過 → fail（阻止寫入）
L4 不通過 → warning（顯示但不阻止寫入）
"""
from __future__ import annotations

import ast
import py_compile
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ── 複雜度計算 ───────────────────────────────────────────────────────────────

# AST 節點類型：每個分支給複雜度 +1
_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.Assert,
    ast.comprehension,
)
_BOOL_OPS = (ast.And, ast.Or)


def _cyclomatic_complexity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """
    計算單一函式的 Cyclomatic Complexity（McCabe 簡化版）。
    基礎分 1，每遇到一個分支節點 +1，每個 BoolOp 的 values-1 額外 +N。
    """
    complexity = 1
    for node in ast.walk(func_node):
        if isinstance(node, _BRANCH_NODES):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # and/or 裡每多一個運算元就多一條路徑
            complexity += len(node.values) - 1
    return complexity


def _extract_function_complexities(source: str) -> dict[str, int]:
    """回傳 {函式名: Cyclomatic Complexity}，語法有錯時回傳 {}。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = _cyclomatic_complexity(node)
    return result


# ── 函式名稱 ─────────────────────────────────────────────────────────────────

def _extract_function_names(source: str) -> set[str]:
    """從 Python 原始碼抓出所有函式定義名稱。語法有錯回傳 {}。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


# ── Import 抽取 ──────────────────────────────────────────────────────────────

def _extract_imports(source: str) -> set[str]:
    """
    抽取所有 import 陳述式的原始行文字（正規化後），
    用來比對「修改前有、修改後消失」的情況。
    只比對模組名稱層級，不要求行文字完全相同
    （例如 `import os` 改成 `from os import path` 算保留）。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # 取頂層模組名（os.path → os）
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


# ── py_compile 驗證 ───────────────────────────────────────────────────────────

def _py_compile_check(source: str, label: str) -> str | None:
    """
    用 py_compile.compile() 做字節碼編譯驗證。
    通過回傳 None；失敗回傳錯誤描述字串。
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as f:
        f.write(source)
        tmp_path = f.name
    try:
        py_compile.compile(tmp_path, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        # 把臨時檔案路徑替換成有意義的 label
        msg = str(e).replace(tmp_path, label)
        return msg
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── QAResult ─────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    passed: bool
    errors: list[str] = field(default_factory=list)    # 阻止寫入
    warnings: list[str] = field(default_factory=list)  # 僅提示

    def __bool__(self) -> bool:
        return self.passed


# ── QAAgent ──────────────────────────────────────────────────────────────────

class QAAgent:
    """
    確定性多層 QA Gate。

    complexity_warn_threshold:
        修改後某函式的 CC 比修改前高出超過此值時發出警告（預設 +5）。
    """

    def __init__(
        self,
        files_dir: Path,
        self_file: str = "",
        complexity_warn_threshold: int = 5,
    ):
        self.files_dir = files_dir
        self.self_file = self_file
        self.complexity_warn_threshold = complexity_warn_threshold

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    def run(self, matches: Sequence[tuple[str, str]]) -> QAResult:
        """
        對所有 (rel_path, new_content) 進行多層檢查。
        只檢查 .py 檔案；跳過工具自身檔案。
        """
        errors: list[str] = []
        warnings: list[str] = []

        for rel_path, new_content in matches:
            rel_path = rel_path.strip()

            # 跳過非 .py 或工具自身
            if not rel_path.endswith(".py"):
                continue
            if self.self_file and Path(rel_path).name == self.self_file:
                continue

            old_path = self.files_dir / rel_path
            old_content: str | None = None
            if old_path.exists():
                try:
                    old_content = old_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

            # L1: AST 語法
            l1_err = self._check_syntax(rel_path, new_content)
            if l1_err:
                errors.extend(l1_err)
                # L2/L3/L4/L5 依賴語法合法，語法不過就跳過
                continue

            # L5: py_compile（比 ast 更嚴格）
            l5_err = self._check_py_compile(rel_path, new_content)
            if l5_err:
                errors.extend(l5_err)
                continue

            if old_content is not None:
                # L2: 函式消失
                errors.extend(self._check_missing_functions(rel_path, old_content, new_content))
                # L3: import 消失
                errors.extend(self._check_missing_imports(rel_path, old_content, new_content))
                # L4: 複雜度上升（警告）
                warnings.extend(self._check_complexity(rel_path, old_content, new_content))

        return QAResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ── 各層實作 ─────────────────────────────────────────────────────────────

    def _check_syntax(self, rel_path: str, content: str) -> list[str]:
        """L1: AST 語法合法性。"""
        try:
            ast.parse(content, filename=rel_path)
            return []
        except SyntaxError as e:
            return [f"[L1 語法] {rel_path}: 行 {e.lineno}: {e.msg}"]

    def _check_py_compile(self, rel_path: str, content: str) -> list[str]:
        """L5: py_compile 字節碼編譯驗證。"""
        err = _py_compile_check(content, rel_path)
        if err:
            return [f"[L5 編譯] {rel_path}: {err}"]
        return []

    def _check_missing_functions(
        self, rel_path: str, old_content: str, new_content: str
    ) -> list[str]:
        """L2: 函式是否無故消失。"""
        old_funcs = _extract_function_names(old_content)
        new_funcs = _extract_function_names(new_content)
        missing = sorted(old_funcs - new_funcs)
        if missing:
            return [
                f"[L2 函式消失] {rel_path}: 以下函式修改後消失，"
                f"可能是模型只回傳部分內容: {', '.join(missing)}"
            ]
        return []

    def _check_missing_imports(
        self, rel_path: str, old_content: str, new_content: str
    ) -> list[str]:
        """L3: 必要 import 是否無故消失。"""
        old_modules = _extract_imports(old_content)
        new_modules = _extract_imports(new_content)
        missing = sorted(old_modules - new_modules)
        if missing:
            return [
                f"[L3 import 消失] {rel_path}: 以下模組在修改前有 import，"
                f"修改後消失: {', '.join(missing)}"
            ]
        return []

    def _check_complexity(
        self, rel_path: str, old_content: str, new_content: str
    ) -> list[str]:
        """L4: Cyclomatic Complexity 是否明顯上升（警告，不擋寫入）。"""
        old_cc = _extract_function_complexities(old_content)
        new_cc = _extract_function_complexities(new_content)
        warnings: list[str] = []
        for func_name, new_val in new_cc.items():
            old_val = old_cc.get(func_name, 0)
            delta = new_val - old_val
            if delta > self.complexity_warn_threshold:
                warnings.append(
                    f"[L4 複雜度] {rel_path}: `{func_name}` CC "
                    f"{old_val} → {new_val} (+{delta}，超過閾值 {self.complexity_warn_threshold})"
                )
        return warnings


# ── 向後相容（供 assistant.py 直接 import 的函式） ────────────────────────────

def _extract_function_names_compat(content: str) -> set[str]:
    """保持和 assistant.py 原有 _extract_function_names 相同的介面。"""
    return _extract_function_names(content)


def _check_missing_functions_compat(
    rel_path: str, new_content: str, files_dir: Path, self_file: str = ""
) -> list[str]:
    """保持和 assistant.py 原有 _check_missing_functions 相同的介面。"""
    target = files_dir / rel_path
    if not target.exists():
        return []
    old_content = target.read_text(encoding="utf-8", errors="replace")
    agent = QAAgent(files_dir=files_dir, self_file=self_file)
    return agent._check_missing_functions(rel_path, old_content, new_content)
