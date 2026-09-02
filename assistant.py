#!/usr/bin/env python3
import subprocess
import re
import sys
import termios
import json
import math
from pathlib import Path
from datetime import datetime
import ollama
# 事實查核用(選項 9 進階查核),非必要但強烈建議安裝:
#   pip install transformers torch
# 首次查核時會自動下載 NLI_MODEL 指定的模型(約數百 MB),請保持網路暢通。

# === 資料夾結構 ===
# ./agent.py            <- 本程式
# ./prompts/             <- prompt 範本檔(prompt_code.txt / prompt_data.txt)
# ./files/                <- 要被索引 / 打包 / 修改的目標檔案(.py / .csv)
FOLDER = Path(".").resolve()          # 主程式所在資料夾(輸出檔、索引檔、備份都放這裡)
PROMPT_DIR = FOLDER / "prompts"       # prompt 範本資料夾
FILES_DIR = FOLDER / "files"          # 目標檔案資料夾(程式碼 / 資料表格都放這裡)

DEFAULT_EXT = [".py"]        # 程式碼索引專用副檔名
DATA_EXT = [".csv"]          # 資料表格索引專用副檔名(跟程式碼分開,避免混在一起)
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
EMBED_MODEL = "nomic-embed-text"  # 需先用 `ollama pull nomic-embed-text` 下載
RAG_INDEX_FILE = "rag_index.json"           # 程式碼索引檔(存在 FOLDER 底下)
RAG_DATA_INDEX_FILE = "rag_data_index.json"  # 資料表格索引檔(獨立檔案,不會互相覆蓋)
PROMPT_CODE_FILE = PROMPT_DIR / "prompt_code.txt"   # 程式碼問答模式的 prompt 範本
PROMPT_DATA_FILE = PROMPT_DIR / "prompt_data.txt"   # 資料表格問答模式的 prompt 範本
RAG_CHUNK_LINES = 60      # 每個索引片段的行數
RAG_CHUNK_OVERLAP = 10    # 片段之間重疊的行數,避免切在函式中間找不到上下文
RAG_TOP_K = 8             # 查詢時取最相關的幾個片段
RAG_MIN_SIMILARITY = 0.4  # 相似度低於此門檻的片段不採用,避免湊數稀釋上下文
RAG_LOW_CONFIDENCE_MAX_SIM = 0.5  # 若本次 top-k 裡最高分都低於這個值,提示回答可信度可能偏低
# 偵測問題裡是否有「路徑式關鍵字」的正則表達式:StageN(不分大小寫,例如 stage3、Stage3、STAGE 3)
_STAGE_KEYWORD_RE = re.compile(r'stage\s*([1-9])', re.IGNORECASE)
# 問題關鍵字 → 預期應該被檢索到的檔名關鍵字。用來提示「這類問題通常要看某個檔案,
# 但這次沒有檢索到」,幫助你判斷回答可信度,純粹是啟發式提示,不是嚴謹規則。
RAG_COVERAGE_HINTS = {
    "model": ["models.py"], "欄位": ["models.py"], "資料庫": ["models.py"],
    "url": ["urls.py"], "路由": ["urls.py"], "路徑": ["urls.py"],
    "serializer": ["serializers.py"], "序列化": ["serializers.py"],
    "admin": ["admin.py"], "後台": ["admin.py"],
    "chatbot": ["chatbot.py"], "聊天機器人": ["chatbot.py"],
    "setting": ["settings.py"], "設定檔": ["settings.py"], "時區": ["settings.py"], "語系": ["settings.py"],
}
# === 三層事實查核設定 ===
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_ENTAIL_THRESHOLD = 0.5
PATTERN = re.compile(r"##### FILE_START: (.+?) #####\n(.*?)\n##### FILE_END #####", re.DOTALL)
SELF_FILE = Path(__file__).resolve().name


def _ensure_dirs():
    """啟動時確保 prompts/ 與 files/ 資料夾存在。"""
    created = []
    for d in (PROMPT_DIR, FILES_DIR):
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    if created:
        print("已自動建立以下資料夾:")
        for d in created:
            print(f"  - {d}")
        print("請把要索引/修改的 .py、.csv 檔放進 files/,")
        print("把 prompt_code.txt、prompt_data.txt 放進 prompts/。\n")


def _flush_stdin():
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def _files(ext_list, folder=None):
    """掃描目標檔案資料夾(預設 FILES_DIR)裡符合副檔名的檔案。"""
    folder = folder or FILES_DIR
    files = []
    for ext in ext_list:
        files += sorted(folder.rglob(f"*{ext}"))
    return [f for f in files if ".backup" not in str(f) and "__pycache__" not in str(f) and f.name != SELF_FILE]


def _copy_to_clipboard(text: str) -> bool:
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        return True
    except Exception:
        return False


def _read_clipboard() -> str:
    try:
        result = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        print(f"讀取剪貼簿失敗:{e}(請確認已安裝 xclip)")
        return ""


_IGNORE_NAMES = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".idea", ".vscode"}


def _is_ignored(p: Path) -> bool:
    if p.name in _IGNORE_NAMES:
        return True
    if p.name.startswith(".backup_"):
        return True
    if p.name == SELF_FILE:
        return True
    return False


def _print_tree(folder: Path, prefix: str = ""):
    try:
        entries = sorted(
            [p for p in folder.iterdir() if not _is_ignored(p)],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        print(f"{prefix}└── (無權限讀取)")
        return
    if not entries:
        print(f"{prefix}(空資料夾)")
        return
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        print(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _print_tree(entry, prefix + extension)


def show_structure():
    _flush_stdin()
    path_input = input(f"請輸入要顯示的資料夾路徑(留空使用目標檔案資料夾 {FILES_DIR}): ").strip()
    if path_input:
        target_folder = Path(path_input).expanduser().resolve()
    else:
        target_folder = FILES_DIR
    if not target_folder.exists():
        print(f"路徑不存在:{target_folder}")
        return
    if not target_folder.is_dir():
        print(f"這不是一個資料夾:{target_folder}")
        return
    print(f"\n=== 資料夾結構:{target_folder} ===")
    print(f"{target_folder.name}/")
    _print_tree(target_folder)
    print()


def collect_for_claude():
    blocks = []
    for f in _files(DEFAULT_EXT):
        rel = f.relative_to(FILES_DIR)
        content = f.read_text(encoding="utf-8", errors="replace")
        blocks.append(f"##### FILE_START: {rel} #####\n{content}\n##### FILE_END #####")
    text = "\n\n".join(blocks)
    Path("collected.txt").write_text(text, encoding="utf-8")
    ok = _copy_to_clipboard(text)
    print(f"\n已整理 {len(text)} 字元 → collected.txt")
    print("已複製到剪貼簿,可直接貼給 Claude" if ok else "請手動打開 collected.txt 複製")


def summarize_with_local_model():
    blocks = []
    for f in _files(DEFAULT_EXT):
        rel = f.relative_to(FILES_DIR)
        content = f.read_text(encoding="utf-8", errors="replace")
        print(f"分析中:{rel} ...")
        prompt = f"""請用繁體中文簡短條列:1)整體用途 2)定義的函式與各自作用 3)明顯的TODO或寫死參數
檔案:{rel}
內容:
{content}"""
        resp = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
        blocks.append(f"### {rel}\n{resp['message']['content']}")
    text = "\n\n".join(blocks)
    Path("summary.txt").write_text(text, encoding="utf-8")
    ok = _copy_to_clipboard(text)
    print(f"\n已存到 summary.txt")
    print("已複製到剪貼簿" if ok else "請手動打開 summary.txt 複製")


def _apply_matches(matches):
    print(f"\n偵測到 {len(matches)} 個檔案異動:")
    for rel_path, _ in matches:
        flag = "  ⚠️ 將被跳過(工具自身檔案,受保護)" if Path(rel_path.strip()).name == SELF_FILE else ""
        print(f"  - {rel_path.strip()}{flag}")
    _flush_stdin()
    confirm = input("\n確認套用嗎?(y/n): ")
    if confirm.lower() != "y":
        print("已取消")
        return
    backup_dir = FILES_DIR / f".backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup_dir.mkdir(exist_ok=True)
    for rel_path, content in matches:
        rel_path = rel_path.strip()
        if Path(rel_path).name == SELF_FILE:
            print(f"已跳過(保護工具自身):{rel_path}")
            continue
        target = FILES_DIR / rel_path
        if target.exists():
            bkp = backup_dir / rel_path
            bkp.parent.mkdir(parents=True, exist_ok=True)
            bkp.write_text(target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"已更新:{rel_path}")
    print(f"備份於:{backup_dir}")


def apply_from_clipboard():
    text = _read_clipboard()
    matches = PATTERN.findall(text)
    if not matches:
        print("剪貼簿內容沒有符合 FILE_START/FILE_END 格式,取消套用")
        return
    _apply_matches(matches)


def auto_edit_with_local_model():
    print("請先把修改指令『複製』到剪貼簿(不要直接貼在這個終端機視窗裡),完成後回到這裡按一下 Enter 繼續...")
    input()
    _flush_stdin()
    instruction = _read_clipboard().strip()
    if not instruction:
        print("剪貼簿是空的,取消操作")
        return
    if "FILE_START" in instruction or "FILE_END" in instruction:
        print("警告:指令內容包含 FILE_START/FILE_END 字樣,可能干擾解析,建議修改指令內容後再試")
        return
    print(f"\n讀到的指令:\n{instruction}\n")
    _flush_stdin()
    confirm = input("確認用這段指令執行嗎?(y/n): ")
    if confirm.lower() != "y":
        print("已取消")
        return
    blocks = []
    for f in _files(DEFAULT_EXT):
        rel = f.relative_to(FILES_DIR)
        content = f.read_text(encoding="utf-8", errors="replace")
        blocks.append(f"##### FILE_START: {rel} #####\n{content}\n##### FILE_END #####")
    files_text = "\n\n".join(blocks)
    prompt = f"""你是程式碼修改助手。請根據下方【指示開始】到【指示結束】之間的內容修改檔案,並【務必】用完全相同的 FILE_START/FILE_END 格式回覆每個檔案(不論有無修改),不要加任何額外說明。
【指示開始】
{instruction}
【指示結束】
以下是檔案內容,只有 FILE_START/FILE_END 標記之間的內容才是檔案內容,不要把上面的指示誤認為是檔案內容:
{files_text}
"""
    print(f"傳送給本地模型修改中(prompt 共 {len(prompt)} 字元),請稍候...")
    Path("debug_last_prompt.txt").write_text(prompt, encoding="utf-8")
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        options={"num_ctx": 16384},
    )
    result = resp['message']['content']
    matches = PATTERN.findall(result)
    if not matches:
        print("模型回覆格式不符,無法套用。前 500 字:")
        print(result[:500])
        return
    _apply_matches(matches)


def extract_relevant_code_with_local_model():
    print("請先把 Claude 提出的『需要修改的部份』說明複製到剪貼簿(不要直接貼在這個終端機視窗裡),完成後回到這裡按一下 Enter 繼續...")
    input()
    _flush_stdin()
    instruction = _read_clipboard().strip()
    if not instruction:
        print("剪貼簿是空的,取消操作")
        return
    if "FILE_START" in instruction or "FILE_END" in instruction:
        print("警告:內容包含 FILE_START/FILE_END 字樣,可能干擾解析,建議修改內容後再試")
        return
    print(f"\n讀到的需求說明:\n{instruction}\n")
    _flush_stdin()
    confirm = input("確認用這段說明判斷相關檔案嗎?(y/n): ")
    if confirm.lower() != "y":
        print("已取消")
        return
    blocks = []
    for f in _files(DEFAULT_EXT):
        rel = f.relative_to(FILES_DIR)
        content = f.read_text(encoding="utf-8", errors="replace")
        blocks.append(f"##### FILE_START: {rel} #####\n{content}\n##### FILE_END #####")
    files_text = "\n\n".join(blocks)
    prompt = f"""你是程式碼分析助手。請根據下方【需求開始】到【需求結束】之間描述的修改需求,從下面提供的檔案中判斷哪些檔案「需要被修改」。
【需求開始】
{instruction}
【需求結束】
以下是目前所有檔案的完整內容,只有 FILE_START/FILE_END 標記之間的內容才是檔案內容,不要把上面的需求說明誤認為是檔案內容:
{files_text}
請【只】針對你判斷需要修改的檔案,用完全相同的 FILE_START/FILE_END 格式回覆這些檔案的「目前完整內容」(原封不動,不要做任何修改或摘要),不要加任何額外說明,也不要包含不相關的檔案。
"""
    print(f"傳送給本地模型判斷相關檔案中(prompt 共 {len(prompt)} 字元),請稍候...")
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        options={"num_ctx": 16384},
    )
    result = resp['message']['content']
    matches = PATTERN.findall(result)
    if not matches:
        print("模型回覆格式不符,無法整理。前 500 字:")
        print(result[:500])
        return
    text = "\n\n".join(
        f"##### FILE_START: {rel.strip()} #####\n{content}\n##### FILE_END #####"
        for rel, content in matches
    )
    Path("relevant_files.txt").write_text(text, encoding="utf-8")
    ok = _copy_to_clipboard(text)
    print(f"\n已判斷出 {len(matches)} 個相關檔案 → relevant_files.txt")
    for rel_path, _ in matches:
        print(f"  - {rel_path.strip()}")
    print("已複製到剪貼簿,可貼給 Claude 取得詳細修改步驟" if ok else "請手動打開 relevant_files.txt 複製")


def review_with_claude():
    print("正在打包目前(修改後)的完整程式碼,準備貼給 Claude 檢查...")
    collect_for_claude()


def _chunk_text(text, chunk_lines=RAG_CHUNK_LINES, overlap_lines=RAG_CHUNK_OVERLAP):
    lines = text.splitlines()
    if not lines:
        return []
    step = max(chunk_lines - overlap_lines, 1)
    chunks = []
    i = 0
    while i < len(lines):
        start, end = i, min(i + chunk_lines, len(lines))
        piece = "\n".join(lines[start:end])
        if piece.strip():
            chunks.append((start + 1, end, piece))
        if end >= len(lines):
            break
        i += step
    return chunks


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_rag_index(ext_list, index_file, mode_label):
    """
    建立/更新 RAG 索引(泛用版,可用於程式碼或資料表格)。
    把 files/ 資料夾內符合 ext_list 副檔名的檔案切成片段,逐片段呼叫 Ollama 的
    embedding 模型算出向量,存成 index_file(放在主程式資料夾),供查詢時使用。
    """
    files = _files(ext_list)
    if not files:
        print(f"在 {FILES_DIR} 裡找不到可索引的檔案(副檔名:{', '.join(ext_list)})")
        return
    print(f"開始建立【{mode_label}】RAG 索引,共 {len(files)} 個檔案,使用 embedding 模型:{EMBED_MODEL}")
    index = []
    fail_count = 0
    for f in files:
        rel = str(f.relative_to(FILES_DIR))
        content = f.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(content)
        print(f"  {rel}: {len(chunks)} 個片段")
        for start, end, piece in chunks:
            embed_input = f"檔案路徑: {rel}\n\n{piece}"
            try:
                resp = ollama.embeddings(model=EMBED_MODEL, prompt=embed_input)
                embedding = resp.get("embedding")
            except Exception as e:
                print(f"    ⚠️ 嵌入失敗({rel} 行 {start}-{end}):{e}")
                fail_count += 1
                continue
            if not embedding:
                fail_count += 1
                continue
            index.append({
                "file": rel,
                "start_line": start,
                "end_line": end,
                "text": piece,
                "embedding": embedding,
            })
    if not index:
        print(f"\n索引建立失敗,沒有任何片段成功嵌入。請確認已執行:ollama pull {EMBED_MODEL}")
        return
    index_path = FOLDER / index_file
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"\n已建立【{mode_label}】RAG 索引:{len(index)} 個片段(失敗 {fail_count} 個)→ {index_path}")


def build_code_index():
    """選項 8:建立/更新「程式碼」RAG 索引(只吃 files/ 底下的 .py)。"""
    build_rag_index(DEFAULT_EXT, RAG_INDEX_FILE, "程式碼")


def build_data_index():
    """選項 9:建立/更新「資料表格」RAG 索引(只吃 files/ 底下的 .csv)。"""
    build_rag_index(DATA_EXT, RAG_DATA_INDEX_FILE, "資料表格")


def _load_rag_index(index_file):
    index_path = FOLDER / index_file
    if not index_path.exists():
        return None
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"讀取索引失敗:{e}")
        return None


def _load_prompt_template(path) -> str:
    """
    讀取 prompts/ 資料夾底下的 prompt 範本檔(prompt_code.txt / prompt_data.txt)。
    範本內用 {question} 與 {context_text} 這兩個佔位字串,讀取後用
    str.format() 代入實際內容。獨立成外部檔案的目的是讓 code/data
    兩種模式各自的措辭可以直接編輯 txt 檔調整,不用改動程式本身。
    """
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(f"找不到 prompt 範本檔案:{template_path}")
    return template_path.read_text(encoding="utf-8")


def _check_coverage_hints(question: str, retrieved_files):
    warnings = []
    q_lower = question.lower()
    retrieved_lower = [f.lower() for f in retrieved_files]
    seen_expected = set()
    for keyword, expected_files in RAG_COVERAGE_HINTS.items():
        if keyword not in q_lower:
            continue
        for expected in expected_files:
            if expected in seen_expected:
                continue
            seen_expected.add(expected)
            if not any(expected in f for f in retrieved_lower):
                warnings.append(f"問題疑似跟「{expected}」有關,但這次檢索到的片段裡沒有這個檔案")
    return warnings


# === 規則式查核(不靠模型,只靠正則表達式) ===
_NUM_RE = re.compile(r'-?\d+\.\d+|-?\d+')
_DATE_RE = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月(?:\d{1,2}日)?|\d{4}年')
_BACKTICK_RE = re.compile(r'`([^`\n]{1,60})`')
_FUNC_CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{1,40})\s*\(')
_SNAKE_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b')
_PY_FILE_RE = re.compile(r'\b[\w./\\-]+\.py\b')


def _extract_code_entities(text: str):
    ents = set()
    ents.update(m.strip() for m in _BACKTICK_RE.findall(text) if m.strip())
    ents.update(_FUNC_CALL_RE.findall(text))
    ents.update(_SNAKE_RE.findall(text))
    return ents


def _rule_based_check(answer: str, context_text: str, retrieved_files, question: str = "", mode: str = "code"):
    """
    規則式查核。mode="code" 時比照原本邏輯(數字/日期/程式碼識別字/.py檔名 都查);
    mode="data" 時只查數字/日期,略過程式碼識別字與 .py 檔名檢查——
    因為 CSV 資料表格裡的欄位值(如訂單編號 A1001)不是程式碼識別字,
    用程式碼的規則去比對只會產生無意義的雜訊,反而稀釋真正有用的查核訊號。
    """
    issues = []
    ans_numbers = set(_NUM_RE.findall(answer))
    missing_numbers = sorted(n for n in ans_numbers if n not in context_text)
    if missing_numbers:
        question_numbers = set(_NUM_RE.findall(question))
        from_question = sorted(n for n in missing_numbers if n in question_numbers)
        genuinely_new = sorted(n for n in missing_numbers if n not in question_numbers)
        if genuinely_new:
            issues.append(f"回答提到的數字在片段原文中找不到:{', '.join(genuinely_new)}")
        if from_question:
            issues.append(
                f"回答提到的數字在片段原文中找不到,但這些數字也出現在問題裡"
                f"(可能只是複誦門檻值,也可能是誤植成具體事實,建議人工確認):{', '.join(from_question)}"
            )
    ans_dates = set(_DATE_RE.findall(answer))
    missing_dates = sorted(d for d in ans_dates if d not in context_text)
    if missing_dates:
        issues.append(f"回答提到的日期在片段原文中找不到:{', '.join(missing_dates)}")
    if mode == "code":
        ans_entities = _extract_code_entities(answer)
        missing_entities = sorted(e for e in ans_entities if e not in context_text)
        if missing_entities:
            issues.append(f"回答提到的程式碼識別字/函式名在片段原文中找不到:{', '.join(missing_entities)}")
        ans_files = set(_PY_FILE_RE.findall(answer))
        missing_files = sorted(f for f in ans_files if not any(f in rf or rf in f for rf in retrieved_files))
        if missing_files:
            issues.append(f"回答提到的檔案不在這次檢索到的片段來源中:{', '.join(missing_files)}")
    return issues


def _sentence_missing_tokens(sentence: str, context_text: str, retrieved_files, mode: str = "code") -> list:
    missing = []
    missing += [n for n in set(_NUM_RE.findall(sentence)) if n not in context_text]
    missing += [d for d in set(_DATE_RE.findall(sentence)) if d not in context_text]
    if mode == "code":
        missing += [e for e in _extract_code_entities(sentence) if e not in context_text]
        missing += [f for f in set(_PY_FILE_RE.findall(sentence))
                    if not any(f in rf or rf in f for rf in retrieved_files)]
    return missing


# === NLI 查核(獨立架構模型,不同於生成模型) ===
_nli_pipeline = None


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is False:
        return None
    if _nli_pipeline is not None:
        return _nli_pipeline
    try:
        from transformers import pipeline
    except ImportError:
        print("⚠️ 尚未安裝 transformers,NLI 查核將略過。請執行:pip install transformers torch")
        _nli_pipeline = False
        return None
    print(f"首次使用 NLI 查核,載入模型 {NLI_MODEL}(需要網路下載,請稍候)...")
    try:
        _nli_pipeline = pipeline("text-classification", model=NLI_MODEL, top_k=None)
    except Exception as e:
        print(f"⚠️ NLI 模型載入失敗,查核將略過:{e}")
        _nli_pipeline = False
        return None
    return _nli_pipeline


def _nli_check(sentences, context_items):
    nli = _get_nli_pipeline()
    if nli is None:
        return None
    results = []
    for sent in sentences:
        best_score, best_src = 0.0, None
        for item in context_items:
            try:
                scores = nli({"text": item["text"], "text_pair": sent}, truncation=True)
            except Exception:
                continue
            entail = next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)
            if entail > best_score:
                best_score, best_src = entail, f"{item['file']} 行{item['start_line']}-{item['end_line']}"
        results.append({"sentence": sent, "score": best_score, "source": best_src})
    return results


def _run_fact_checks(context_items, context_text, retrieved_files, answer, question: str = "", mode: str = "code"):
    print("\n=== 規則式查核(數字 / 日期" + (" / 程式碼識別字 / 檔名" if mode == "code" else "") + ")===")
    rule_issues = _rule_based_check(answer, context_text, retrieved_files, question=question, mode=mode)
    if rule_issues:
        for issue in rule_issues:
            print(f"  ⚠️ {issue}")
    else:
        print("  未發現對不上片段的狀況。")
    print(f"\n=== NLI 事實查核(獨立模型:{NLI_MODEL})===")
    sentences = [s.strip() for s in re.split(r'(?<=[。!?\n])', answer) if s.strip() and len(s.strip()) >= 4]
    nli_results = _nli_check(sentences, context_items)
    conflict_count = 0
    if nli_results is None:
        print("  已略過(未安裝或載入失敗,可執行:pip install transformers torch)")
    else:
        for r in nli_results:
            r["rule_missing"] = _sentence_missing_tokens(r["sentence"], context_text, retrieved_files, mode=mode)
            r["conflict"] = bool(r["rule_missing"]) and r["score"] >= NLI_ENTAIL_THRESHOLD
            if r["conflict"]:
                conflict_count += 1
                flag = "🚨 查核衝突(規則式抓到問題,但NLI判定有依據)"
            elif r["score"] >= NLI_ENTAIL_THRESHOLD:
                flag = "✅"
            else:
                flag = "⚠️ 疑似缺乏依據"
            print(f"  {flag} [{r['score']:.2f}] {r['sentence']}")
            if r["source"]:
                print(f"        最佳佐證來源:{r['source']}")
            if r["rule_missing"]:
                print(f"        規則式查核發現片段裡沒有:{', '.join(r['rule_missing'])}")
        if conflict_count:
            print(f"\n  🚨 共 {conflict_count} 句查核結果衝突,建議優先人工確認這幾句。")
    print()
    return rule_issues, nli_results


def _build_report_text(question: str, answer: str, rule_issues, nli_results, final_summary: str = None) -> str:
    parts = [f"【問題】\n{question}", f"\n【回答】\n{answer}"]
    parts.append("\n【規則式查核】")
    if rule_issues:
        parts.extend(f"⚠️ {issue}" for issue in rule_issues)
    else:
        parts.append("未發現對不上片段的狀況。")
    parts.append(f"\n【NLI 事實查核(獨立模型:{NLI_MODEL})】")
    if nli_results is None:
        parts.append("已略過(未安裝或載入失敗,可執行:pip install transformers torch)")
    else:
        conflict_count = 0
        for r in nli_results:
            if r.get("conflict"):
                conflict_count += 1
                flag = "🚨 查核衝突(規則式抓到問題,但NLI判定有依據)"
            elif r["score"] >= NLI_ENTAIL_THRESHOLD:
                flag = "✅"
            else:
                flag = "⚠️ 疑似缺乏依據"
            line = f"{flag} [{r['score']:.2f}] {r['sentence']}"
            if r["source"]:
                line += f"\n    最佳佐證來源:{r['source']}"
            if r.get("rule_missing"):
                line += f"\n    規則式查核發現片段裡沒有:{', '.join(r['rule_missing'])}"
            parts.append(line)
        if conflict_count:
            parts.append(f"\n🚨 共 {conflict_count} 句查核結果衝突,建議優先人工確認這幾句。")
    if final_summary:
        parts.append("\n【最終彙整總結(純程式邏輯依查核結果分組,未經任何 LLM 二次生成)】")
        parts.append(final_summary)
    return "\n".join(parts)


def _summarize_final_answer(question: str, answer: str, rule_issues, nli_results) -> str:
    """
    最終彙整,改用【純程式邏輯組裝,不呼叫任何 LLM】。
    原本這裡是再叫一次 LLM_MODEL 幫忙把「已確認內容」跟「缺乏依據內容」分成
    兩堆,但這個任務本質上只是「依照規則式查核+NLI查核已經算好的布林值做條件
    篩選」,不需要語言理解能力;實測發現讓 7B 小模型執行這種互斥篩選指令時,
    容易把同一句話同時塞進「已確認」跟「缺乏依據」兩邊,產生自相矛盾的總結。
    改成直接讀 nli_results 裡每句話算好的 score/conflict 分組,可以完全避免
    這類錯誤 —— 不是「比較不會錯」,是這個步驟不再存在讓它出錯的空間。
    """
    if nli_results is None:
        if rule_issues:
            parts = [
                "(NLI 查核未啟用,無法逐句判定,以下為原始回答,請自行比對規則式查核結果)",
                "",
                answer,
                "",
                "⚠️ 規則式查核發現以下問題,建議人工確認:",
            ]
            parts.extend(f"- {issue}" for issue in rule_issues)
            return "\n".join(parts)
        return answer
    confirmed_sentences = []
    flagged_sentences = []
    for r in nli_results:
        is_flagged = bool(r.get("conflict")) or r["score"] < NLI_ENTAIL_THRESHOLD
        (flagged_sentences if is_flagged else confirmed_sentences).append(r["sentence"])
    parts = []
    if confirmed_sentences:
        parts.append("".join(confirmed_sentences))
    else:
        parts.append("(這次回答沒有通過查核的內容,以下全部屬於缺乏依據或查核衝突,請人工確認)")
    if flagged_sentences:
        parts.append("\n⚠️ 以下內容缺乏明確依據或查核衝突,建議人工確認:")
        parts.extend(f"- {s}" for s in flagged_sentences)
    if rule_issues:
        parts.append("\n⚠️ 規則式查核在整體回答中另外發現以下問題:")
        parts.extend(f"- {issue}" for issue in rule_issues)
    return "\n".join(parts)


def query_rag(index_file: str, mode: str, mode_label: str):
    """
    泛用版查詢函式,依 mode 分成「code」或「data」。
    mode="code":維持原本針對程式碼設計的行為(Stage 路徑過濾、程式碼查核規則)。
    mode="data":跳過 Stage 路徑過濾與程式碼專用的規則式查核,適合資料表格。
    """
    index = _load_rag_index(index_file)
    if not index:
        print(f"尚未建立【{mode_label}】RAG 索引,請先執行對應的建立索引選項")
        return
    _flush_stdin()
    question = input("請輸入你想查詢的問題: ").strip()
    if not question:
        print("問題是空的,取消操作")
        return
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=question)
        q_embedding = resp.get("embedding")
    except Exception as e:
        print(f"問題嵌入失敗:{e}")
        return
    if not q_embedding:
        print("問題嵌入失敗")
        return
    scored_all = sorted(
        ((_cosine_sim(q_embedding, item["embedding"]), item) for item in index),
        key=lambda x: x[0],
        reverse=True,
    )
    path_keywords = []
    matched = []
    if mode == "code":
        stage_matches = _STAGE_KEYWORD_RE.findall(question)
        path_keywords = [f"stage{n}" for n in stage_matches]
        if path_keywords:
            matched = [(sim, item) for sim, item in scored_all
                       if any(kw in item["file"].lower() for kw in path_keywords)]
            print(f"  (偵測到路徑關鍵字 {path_keywords},此資料夾底下共 {len(matched)} 個片段,只使用這些片段,不補其他檔案)")
    if path_keywords and matched:
        scored = matched
    elif path_keywords and not matched:
        print(f"  ⚠️ 索引裡完全找不到符合 {path_keywords} 的片段,退回一般語意搜尋。")
        scored = scored_all
    else:
        scored = scored_all
    if path_keywords and matched:
        filtered = scored
    else:
        filtered = [(sim, item) for sim, item in scored if sim >= RAG_MIN_SIMILARITY]
    top_k = filtered[:RAG_TOP_K] if filtered else scored[:RAG_TOP_K]
    print("\n=== 最相關的片段 ===")
    context_blocks = []
    retrieved_files = []
    for sim, item in top_k:
        below_flag = "" if sim >= RAG_MIN_SIMILARITY else "  ⚠️ 低於相似度門檻,僅供備援"
        print(f"  [{sim:.3f}] {item['file']} (行 {item['start_line']}-{item['end_line']}){below_flag}")
        context_blocks.append(f"# 檔案: {item['file']} (行 {item['start_line']}-{item['end_line']})\n{item['text']}")
        retrieved_files.append(item["file"])
    context_text = "\n\n".join(context_blocks)
    top_sim = top_k[0][0] if top_k else 0.0
    if top_sim < RAG_LOW_CONFIDENCE_MAX_SIM:
        print(f"\n⚠️ 提示:這次最高相似度只有 {top_sim:.3f},檢索到的片段跟問題關聯度偏低,回答可信度建議打折扣。")
    if mode == "code":
        coverage_warnings = _check_coverage_hints(question, retrieved_files)
        if coverage_warnings:
            print("\n⚠️ 檢索覆蓋度提示:")
            for w in coverage_warnings:
                print(f"  - {w}")
    template_file = PROMPT_CODE_FILE if mode == "code" else PROMPT_DATA_FILE
    try:
        template = _load_prompt_template(template_file)
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        return
    try:
        prompt = template.format(question=question, context_text=context_text)
    except (KeyError, IndexError) as e:
        print(f"⚠️ prompt 範本檔案 {template_file} 格式有誤(佔位字串應為 {{question}} 與 {{context_text}}):{e}")
        return
    print("\n查詢本地模型中,請稍候...")
    resp = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    answer = resp['message']['content']
    print(f"\n=== 回答 ===\n{answer}\n")
    context_items = [item for _, item in top_k]
    rule_issues, nli_results = _run_fact_checks(context_items, context_text, retrieved_files, answer, question=question, mode=mode)
    final_summary = _summarize_final_answer(question, answer, rule_issues, nli_results)
    print(f"\n=== 最終彙整總結 ===\n{final_summary}\n")
    report_text = _build_report_text(question, answer, rule_issues, nli_results, final_summary)
    _flush_stdin()
    save = input("是否要複製「回答＋查核結果＋最終總結」到剪貼簿?(y/n): ").strip().lower()
    if save == "y":
        ok = _copy_to_clipboard(report_text)
        print("已複製到剪貼簿(含查核結果與總結)" if ok else "複製失敗")


def query_code_rag():
    """選項 10:針對「程式碼」RAG 索引提問。"""
    query_rag(RAG_INDEX_FILE, "code", "程式碼")


def query_data_rag():
    """選項 11:針對「資料表格」RAG 索引提問。"""
    query_rag(RAG_DATA_INDEX_FILE, "data", "資料表格")


def menu():
    _ensure_dirs()
    while True:
        print(f"""
========== Agent Project 助理 ==========
(目標檔案資料夾: {FILES_DIR})
(prompt 範本資料夾: {PROMPT_DIR})
1. 顯示資料夾結構(可指定路徑,留空為 files/ 資料夾)
2. 打包程式碼給 Claude(複製到剪貼簿)
3. 用本地模型摘要整理(複製到剪貼簿)
4. 從剪貼簿套用修改(Claude 回覆貼回後用)
5. 用本地模型直接修改(貼 Claude 生成的指令)
6. 用本地模型整理需要修改部份的完整程式碼(貼 Claude 說明後用)
7. 修改完把程式貼回 Claude 檢查
8. 建立/更新「程式碼」RAG 索引 (.py)
9. 建立/更新「資料表格」RAG 索引 (.csv)
10. 用 RAG 提問查詢「程式碼」(含規則式+NLI雙重查核 + 純程式彙整,不額外呼叫LLM)
11. 用 RAG 提問查詢「資料表格」(含規則式+NLI雙重查核 + 純程式彙整,不額外呼叫LLM)
0. 離開
=========================================
""")
        _flush_stdin()
        choice = input("選擇: ").strip()
        if choice == "1":
            show_structure()
        elif choice == "2":
            collect_for_claude()
        elif choice == "3":
            summarize_with_local_model()
        elif choice == "4":
            apply_from_clipboard()
        elif choice == "5":
            auto_edit_with_local_model()
        elif choice == "6":
            extract_relevant_code_with_local_model()
        elif choice == "7":
            review_with_claude()
        elif choice == "8":
            build_code_index()
        elif choice == "9":
            build_data_index()
        elif choice == "10":
            query_code_rag()
        elif choice == "11":
            query_data_rag()
        elif choice == "0":
            sys.exit(0)
        else:
            print("無效選項")


if __name__ == "__main__":
    menu()
