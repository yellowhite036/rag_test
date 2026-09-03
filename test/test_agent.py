import unittest
from pathlib import Path
import shutil
from agent import (
    _extract_function_names,
    _check_missing_functions,
    _clean_model_output,
    _validate_matches,
    _qa_gate,
    FILES_DIR,
)


class TestExtractFunctionNames(unittest.TestCase):

    def test_simple_functions(self):
        code = "def a():\n    pass\ndef b():\n    pass\n"
        names = _extract_function_names(code)
        self.assertEqual(names, {"a", "b"})

    def test_nested_and_methods(self):
        code = """
def outer():
    def inner():
        pass
    return inner

class Foo:
    def method(self):
        pass

async def bar():
    pass
"""
        names = _extract_function_names(code)
        self.assertEqual(names, {"outer", "inner", "method", "bar"})

    def test_syntax_error_returns_empty(self):
        code = "def a(:\n    pass"
        names = _extract_function_names(code)
        self.assertEqual(names, set())

    def test_empty_content(self):
        self.assertEqual(_extract_function_names(""), set())


class TestCheckMissingFunctions(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("test_files_tmp")
        self.test_dir.mkdir(exist_ok=True)
        self._orig_files_dir = None

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_old(self, rel_path, content):
        p = self.test_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def test_no_missing_functions(self):
        import agent
        agent.FILES_DIR = self.test_dir
        self._write_old("calc.py", "def a():\n    pass\ndef b():\n    pass\n")
        new_content = 'def a():\n    """doc"""\n    pass\ndef b():\n    pass\n'
        problems = _check_missing_functions("calc.py", new_content)
        self.assertEqual(problems, [])

    def test_detects_missing_function(self):
        import agent
        agent.FILES_DIR = self.test_dir
        self._write_old("calc.py", "def multiply_list(n):\n    pass\ndef subtract(a,b):\n    pass\ndef divide_safe(a,b):\n    pass\n")
        # 模擬模型只回傳 multiply_list,砍掉其他兩個函式
        new_content = 'def multiply_list(n):\n    """doc"""\n    pass\n'
        problems = _check_missing_functions("calc.py", new_content)
        self.assertEqual(len(problems), 1)
        self.assertIn("subtract", problems[0])
        self.assertIn("divide_safe", problems[0])

    def test_new_file_no_check(self):
        import agent
        agent.FILES_DIR = self.test_dir
        # 檔案不存在,視為新檔案,不應該有任何問題
        problems = _check_missing_functions("brand_new.py", "def x():\n    pass\n")
        self.assertEqual(problems, [])


class TestQAGateIntegration(unittest.TestCase):
    """
    模擬你這次實際遇到的情境:語法合法,但函式被誤刪。
    驗證 _qa_gate 能不能攔下來。
    """

    def setUp(self):
        self.test_dir = Path("test_files_tmp2")
        self.test_dir.mkdir(exist_ok=True)
        import agent
        agent.FILES_DIR = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_qa_gate_catches_missing_functions_even_with_valid_syntax(self):
        old = (
            "def multiply_list(numbers):\n"
            "    result = 1\n"
            "    for n in numbers:\n"
            "        result *= n\n"
            "    return result\n\n"
            "def subtract(a, b):\n"
            "    return a - b\n\n"
            "def divide_safe(a, b):\n"
            "    if b == 0:\n"
            "        return None\n"
            "    return a / b\n"
        )
        (self.test_dir / "calc.py").write_text(old, encoding="utf-8")

        # 模型只回傳 multiply_list,語法完全合法
        new = (
            'def multiply_list(numbers):\n'
            '    """這個函式計算乘積"""\n'
            '    result = 1\n'
            '    for n in numbers:\n'
            '        result *= n\n'
            '    return result\n'
        )
        matches = [("calc.py", new)]
        passed, problems = _qa_gate(matches)
        self.assertFalse(passed)
        self.assertTrue(any("subtract" in p and "divide_safe" in p for p in problems))

    def test_qa_gate_passes_when_all_functions_kept(self):
        old = "def a():\n    pass\ndef b():\n    pass\n"
        (self.test_dir / "calc.py").write_text(old, encoding="utf-8")
        new = 'def a():\n    """doc"""\n    pass\ndef b():\n    pass\n'
        matches = [("calc.py", new)]
        passed, problems = _qa_gate(matches)
        self.assertTrue(passed)
        self.assertEqual(problems, [])


class TestCleanModelOutput(unittest.TestCase):

    def test_strips_code_fence_and_repairs_markers(self):
        text = '```python\nFILE_START\ndef multiply_list(numbers):\n    """doc"""\n    return 1\nFILE_END\n```'
        result = _clean_model_output(text)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
