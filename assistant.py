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

FOLDER = Path(".").resolve()
DEFAULT_EXT = [".py"]
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
EMBED_MODEL = "nomic-embed-text"  # 需先用 `ollama pull nomic-embed-text` 下載
RAG_INDEX_FILE = "rag_index.json"
RAG_CHUNK_LINES = 60      # 每個索引片段的行數
RAG_CHUNK_OVERLAP = 10    # 片段之間重疊的行數,避免切在函式中間找不到上下文
RAG_TOP_K = 8             # 查詢時取最相關的幾個片段
RAG_MIN_SIMILARITY = 0.4  # 相似度低於此門檻的片段不採用,避免湊數稀釋上下文
RAG_LOW_CONFIDENCE_MAX_SIM = 0.5  # 若本次 top-k 裡最高分都低於這個值,提示回答可信度可能偏低
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
# 1) 生成模型:LLM_MODEL(Qwen2.5,本地執行,成本可控)—— 負責產生回答
# 2) 獨立架構查核模型:NLI_MODEL,刻意選用跟 LLM_MODEL 不同家族、不同訓練資料的
#    DeBERTa-v3 NLI 模型,專門判斷「回答的每一句話」相對於「檢索到的片段」是否
#    構成蘊含(entailment)。跟原本「同一顆模型自己核對自己」不同,查核者跟生成者
#    是兩套獨立的訓練脈絡,比較不會犯一樣的錯又互相掩護。
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_ENTAIL_THRESHOLD = 0.5  # 蘊含機率低於此值,視為「片段中找不到明確依據」
# 3) 規則式查核:不靠模型,直接用正則表達式比對數字/日期/程式碼識別字/檔名
#    是否確實出現在檢索到的片段原文裡,速度最快、最不會誤判,專門抓模型
#    憑空補出來、片段裡根本沒有的具體細節。

PATTERN = re.compile(r"##### FILE_START: (.+?) #####\n(.*?)\n##### FILE_END #####", re.DOTALL)
SELF_FILE = Path(__file__).resolve().name  # 這支工具自己的檔名,永遠排除在可修改清單外

def _flush_stdin():
    """清空終端機殘留的輸入緩衝區。
    若使用者不小心把多行文字直接貼進終端機(而非只複製到系統剪貼簿),
    input() 每次只會讀走一行,其餘行會殘留在緩衝區裡污染後續的
    input() 呼叫(例如確認 y/n、選單選項)。在關鍵輸入前呼叫這個函式
    可以把殘留內容丟掉,避免莫名其妙變成「已取消」或「無效選項」。
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

def _files(ext_list, folder=None):
    folder = folder or FOLDER
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

# 顯示樹狀結構時要忽略的雜訊資料夾/檔案(不論在哪一層都排除)
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
    path_input = input(f"請輸入要顯示的資料夾路徑(留空使用目前資料夾 {FOLDER}): ").strip()
    if path_input:
        target_folder = Path(path_input).expanduser().resolve()
    else:
        target_folder = FOLDER

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
        rel = f.relative_to(FOLDER)
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
        rel = f.relative_to(FOLDER)
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

    backup_dir = FOLDER / f".backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup_dir.mkdir(exist_ok=True)
    for rel_path, content in matches:
        rel_path = rel_path.strip()
        if Path(rel_path).name == SELF_FILE:
            print(f"已跳過(保護工具自身):{rel_path}")
            continue
        target = FOLDER / rel_path
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
    _flush_stdin()  # 若使用者不小心把多行文字貼進終端機,這裡先丟掉殘留內容
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
        rel = f.relative_to(FOLDER)
        content = f.read_text(encoding="utf-8", errors="replace")
        blocks.append(f"##### FILE_START: {rel} #####\n{content}\n##### FILE_END #####")
    files_text = "\n\n".join(blocks)

    # 用明確的邊界標記包住指令,避免多行指令與檔案內容的界線模糊,
    # 導致模型誤判範圍、回覆格式跑掉(這是原本「多行指令會失效」的主因)
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
    """
    選項 6:貼回本地模型,整理出需要修改部份的完整程式碼。

    使用時機:Claude 已經在對話裡提出「哪些地方需要修改」的說明(還沒給
    出詳細修改步驟),把那段說明複製到剪貼簿後執行本選項。
    本地模型會根據這段說明,從所有檔案裡「判斷」哪些檔案跟需求相關,
    並【原封不動】整理出這些檔案目前完整的程式碼(不做任何修改),
    存檔並複製到剪貼簿,方便你把這些相關檔案的完整內容貼回 Claude,
    讓 Claude 針對這些檔案給出詳細修改步驟(對應流程第 5 步)。
    """
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
        rel = f.relative_to(FOLDER)
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
    """
    選項 7:修改完成後,把目前的完整程式碼打包複製到剪貼簿,貼回 Claude 檢查。
    功能與選項 2 相同(都是打包全部檔案),獨立成一個選項只是為了讓
    流程上的語意更清楚:這是「套用修改之後」的檢查步驟。
    """
    print("正在打包目前(修改後)的完整程式碼,準備貼給 Claude 檢查...")
    collect_for_claude()

def _chunk_text(text, chunk_lines=RAG_CHUNK_LINES, overlap_lines=RAG_CHUNK_OVERLAP):
    """把檔案內容依行數切成有重疊的片段,回傳 [(起始行, 結束行, 片段文字), ...]。
    重疊是為了避免函式剛好被切在中間,導致查詢時上下文不完整。"""
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
            chunks.append((start + 1, end, piece))  # 行號從 1 開始
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

def build_rag_index():
    """
    選項 8:建立/更新 RAG 索引。
    把資料夾內所有 .py 檔切成片段,逐片段呼叫 Ollama 的 embedding 模型
    (預設 nomic-embed-text)算出向量,存成 rag_index.json,供選項 9 查詢使用。
    每次執行都會整份重建索引(檔案有異動後重新執行選項 8 即可)。
    """
    files = _files(DEFAULT_EXT)
    if not files:
        print("找不到可索引的檔案")
        return

    print(f"開始建立 RAG 索引,共 {len(files)} 個檔案,使用 embedding 模型:{EMBED_MODEL}")
    index = []
    fail_count = 0
    for f in files:
        rel = str(f.relative_to(FOLDER))
        content = f.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(content)
        print(f"  {rel}: {len(chunks)} 個片段")
        for start, end, piece in chunks:
            try:
                resp = ollama.embeddings(model=EMBED_MODEL, prompt=piece)
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
        print("\n索引建立失敗,沒有任何片段成功嵌入。請確認已執行:ollama pull " + EMBED_MODEL)
        return

    index_path = FOLDER / RAG_INDEX_FILE
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"\n已建立 RAG 索引:{len(index)} 個片段(失敗 {fail_count} 個)→ {index_path}")

def _load_rag_index():
    index_path = FOLDER / RAG_INDEX_FILE
    if not index_path.exists():
        return None
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"讀取索引失敗:{e}")
        return None

def _check_coverage_hints(question: str, retrieved_files):
    """(A) 檢索覆蓋度提示。
    純粹是關鍵字啟發式檢查:問題裡如果出現某些關鍵字(像「model」「欄位」),
    通常代表答案應該在 models.py 之類的檔案裡;如果這次 top-k 檢索到的檔案
    清單裡完全沒有對應的檔名,回傳警告文字,提醒使用者這次回答可能不完整。
    """
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

def _self_check_answer(question: str, context_text: str, answer: str) -> str:
    """(舊版)回答自我核對 —— 已被下方「三層事實查核」(_run_fact_checks)取代,不再預設呼叫。
    保留這個函式是因為它仍然是有效的備用手段,只是缺點很明顯:核對者跟生成者
    是同一顆模型(LLM_MODEL),容易用同一套錯誤邏輯騙過自己,無法算獨立查核。
    如果想比較兩種查核方式的差異,可以在 query_rag() 裡把這個函式的呼叫加回去。
    """
    check_prompt = f"""你是嚴格的事實核對員。請核對下方「回答」裡的每一句具體陳述,是否都能在「原始程式碼片段」裡找到明確依據。

規則:
1. 逐一列出「回答」中缺乏依據、片段裡沒有明確寫出、屬於推測或編造的具體陳述。
2. 如果全部陳述都有明確依據,只回覆:「未發現缺乏依據的陳述。」
3. 不要重新回答問題,只做核對,用繁體中文條列式列出結果。

【問題】
{question}

【原始程式碼片段】
{context_text}

【回答】
{answer}
"""
    resp = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': check_prompt}])
    return resp['message']['content']

# === 規則式查核(不靠模型,只靠正則表達式) ===
_NUM_RE = re.compile(r'-?\d+\.\d+|-?\d+')
_DATE_RE = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月(?:\d{1,2}日)?|\d{4}年')
_BACKTICK_RE = re.compile(r'`([^`\n]{1,60})`')
_FUNC_CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{1,40})\s*\(')
_SNAKE_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b')
_PY_FILE_RE = re.compile(r'\b[\w./\\-]+\.py\b')

def _extract_code_entities(text: str):
    """從文字裡抓出「看起來像程式碼識別字」的 token:反引號標註的內容、
    函式呼叫 foo(...)、snake_case 命名。這些是模型最容易憑印象亂編的細節。
    """
    ents = set()
    ents.update(m.strip() for m in _BACKTICK_RE.findall(text) if m.strip())
    ents.update(_FUNC_CALL_RE.findall(text))
    ents.update(_SNAKE_RE.findall(text))
    return ents

def _rule_based_check(answer: str, context_text: str, retrieved_files):
    """規則式查核:直接比對答案裡出現的數字/日期/程式碼識別字/檔名,
    是否真的能在檢索到的片段原文(或這次檢索到的檔案清單)裡找到。
    不靠任何模型判斷,速度最快、也最不會誤判(片段裡真的有這串文字才會過),
    專門抓「片段裡根本沒有卻被生出來」的具體細節,跟 NLI 查核互補:
    NLI 抓語意上不被支持的陳述,規則式抓具體數值/名稱對不上的陳述。
    """
    issues = []

    ans_numbers = set(_NUM_RE.findall(answer))
    missing_numbers = sorted(n for n in ans_numbers if n not in context_text)
    if missing_numbers:
        issues.append(f"回答提到的數字在片段原文中找不到:{', '.join(missing_numbers)}")

    ans_dates = set(_DATE_RE.findall(answer))
    missing_dates = sorted(d for d in ans_dates if d not in context_text)
    if missing_dates:
        issues.append(f"回答提到的日期在片段原文中找不到:{', '.join(missing_dates)}")

    ans_entities = _extract_code_entities(answer)
    missing_entities = sorted(e for e in ans_entities if e not in context_text)
    if missing_entities:
        issues.append(f"回答提到的程式碼識別字/函式名在片段原文中找不到:{', '.join(missing_entities)}")

    ans_files = set(_PY_FILE_RE.findall(answer))
    missing_files = sorted(f for f in ans_files if not any(f in rf or rf in f for rf in retrieved_files))
    if missing_files:
        issues.append(f"回答提到的檔案不在這次檢索到的片段來源中:{', '.join(missing_files)}")

    return issues

def _sentence_missing_tokens(sentence: str, context_text: str, retrieved_files) -> list:
    """規則式查核的「逐句」版本:回傳這句話裡有哪些數字/日期/識別字/檔名,
    在片段原文(或這次檢索到的檔案清單)裡找不到。
    用途是跟 NLI 的逐句結果做交叉比對——NLI 是語意層級的判斷,容易被「聽起來合理」
    的長片段唬弄過去;規則式查核是字面比對,只要片段裡真的有這個字串就一定找得到,
    兩者對同一句話的判斷若不一致,代表這句話特別需要人工確認。
    """
    missing = []
    missing += [n for n in set(_NUM_RE.findall(sentence)) if n not in context_text]
    missing += [d for d in set(_DATE_RE.findall(sentence)) if d not in context_text]
    missing += [e for e in _extract_code_entities(sentence) if e not in context_text]
    missing += [f for f in set(_PY_FILE_RE.findall(sentence))
                if not any(f in rf or rf in f for rf in retrieved_files)]
    return missing

# === NLI 查核(獨立架構模型,不同於生成模型) ===
_nli_pipeline = None  # 全域快取,避免每次查詢都重新載入模型

def _get_nli_pipeline():
    """延遲載入 NLI 查核模型,避免沒用到選項 9 時拖慢啟動速度。
    刻意選用跟 LLM_MODEL(Qwen 系列)完全不同家族、不同訓練資料的 DeBERTa-v3
    NLI 模型,確保「查核者」跟「生成者」不是同一套訓練脈絡。
    """
    global _nli_pipeline
    if _nli_pipeline is False:
        return None  # 之前載入失敗過,不再重試
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
    """對回答裡的每一句話,分別跟每個檢索到的片段做自然語言推論(NLI),
    取「蘊含(entailment)」機率最高的片段當作最佳佐證來源。
    分數低於 NLI_ENTAIL_THRESHOLD,代表這句話在目前片段裡找不到明確依據,
    可能是模型自己補上去的內容(幻覺)。回傳 None 代表模型未安裝/載入失敗,已略過查核。
    """
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

def _run_fact_checks(context_items, context_text, retrieved_files, answer):
    """整合「規則式查核」+「NLI 查核(獨立模型)」,印出報告。
    取代原本用同一顆生成模型自己核對自己的做法(_self_check_answer 仍保留但不預設呼叫)。
    """
    print("\n=== 規則式查核(數字 / 日期 / 程式碼識別字 / 檔名)===")
    rule_issues = _rule_based_check(answer, context_text, retrieved_files)
    if rule_issues:
        for issue in rule_issues:
            print(f"  ⚠️ {issue}")
    else:
        print("  未發現數字/日期/識別字/檔名對不上片段的狀況。")

    print(f"\n=== NLI 事實查核(獨立模型:{NLI_MODEL})===")
    sentences = [s.strip() for s in re.split(r'(?<=[。！？\n])', answer) if s.strip() and len(s.strip()) >= 4]
    nli_results = _nli_check(sentences, context_items)
    conflict_count = 0
    if nli_results is None:
        print("  已略過(未安裝或載入失敗,可執行:pip install transformers torch)")
    else:
        # 交叉比對:規則式查核揪出這句有找不到的具體內容,但 NLI 卻判定「有依據」時,
        # 代表 NLI 很可能是被同一個泛用片段的語意相似度唬弄過去(NLI 抓語意支持,
        # 不擅長判斷「這個具體字串到底有沒有寫在片段裡」),兩者結論不一致要特別標記出來,
        # 不能讓使用者只看到 NLI 那排綠色勾勾就以為沒問題。
        for r in nli_results:
            r["rule_missing"] = _sentence_missing_tokens(r["sentence"], context_text, retrieved_files)
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

def _build_report_text(question: str, answer: str, rule_issues, nli_results) -> str:
    """把回答跟三層查核的結果組成一份完整文字報告,供複製到剪貼簿/存檔用,
    這樣貼給 Claude 或存下來時,查核警告不會遺失在終端機裡。
    """
    parts = [f"【問題】\n{question}", f"\n【回答】\n{answer}"]

    parts.append("\n【規則式查核(數字 / 日期 / 程式碼識別字 / 檔名)】")
    if rule_issues:
        parts.extend(f"⚠️ {issue}" for issue in rule_issues)
    else:
        parts.append("未發現數字/日期/識別字/檔名對不上片段的狀況。")

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

    return "\n".join(parts)

def query_rag():
    """
    選項 9:輸入問題,查詢專案程式碼並回答。
    把問題轉成向量,跟索引裡每個片段算 cosine 相似度,取相似度高於
    RAG_MIN_SIMILARITY 的片段中,最相關的 RAG_TOP_K 個當上下文,
    交給本地模型(LLM_MODEL)依上下文回答問題。

    附帶自動輔助檢查,幫助判斷回答是否可能有幻覺:
    A. 檢索覆蓋度提示:問題關鍵字對應的檔案這次有沒有被檢索到
    B. 三層事實查核:
       1. 生成模型(LLM_MODEL,成本可控)產生回答
       2. NLI 查核(獨立架構、不同訓練來源的 DeBERTa-v3 模型)逐句判斷語意上有沒有依據
       3. 規則式查核(純正則表達式)比對數字/日期/識別字/檔名是否真的出現在片段裡
    C. 低相似度警告:這次 top-k 分數普遍偏低時提醒可信度打折扣
    """
    index = _load_rag_index()
    if not index:
        print("尚未建立 RAG 索引,請先執行選項 8 建立索引")
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

    scored = sorted(
        ((_cosine_sim(q_embedding, item["embedding"]), item) for item in index),
        key=lambda x: x[0],
        reverse=True,
    )
    # 先用相似度門檻濾掉不夠相關的片段,避免湊數稀釋上下文;
    # 如果濾完一個都不剩,退回用原始 top_k(至少讓模型有東西可看,
    # 由 prompt 規則負責在資訊不足時誠實說找不到答案)。
    filtered = [(sim, item) for sim, item in scored if sim >= RAG_MIN_SIMILARITY]
    top_k = filtered[:RAG_TOP_K] if filtered else scored[:RAG_TOP_K]

    print("\n=== 最相關的程式碼片段 ===")
    context_blocks = []
    retrieved_files = []
    for sim, item in top_k:
        below_flag = "" if sim >= RAG_MIN_SIMILARITY else "  ⚠️ 低於相似度門檻,僅供備援"
        print(f"  [{sim:.3f}] {item['file']} (行 {item['start_line']}-{item['end_line']}){below_flag}")
        context_blocks.append(f"# 檔案: {item['file']} (行 {item['start_line']}-{item['end_line']})\n{item['text']}")
        retrieved_files.append(item["file"])
    context_text = "\n\n".join(context_blocks)

    # (C) 低相似度警告:這次找到的東西整體跟問題關聯度都不高,先提醒使用者。
    top_sim = top_k[0][0] if top_k else 0.0
    if top_sim < RAG_LOW_CONFIDENCE_MAX_SIM:
        print(f"\n⚠️ 提示:這次最高相似度只有 {top_sim:.3f},檢索到的片段跟問題關聯度偏低,回答可信度建議打折扣。")

    # (A) 檢索覆蓋度提示:問題關鍵字對應的檔案這次有沒有被檢索到。
    coverage_warnings = _check_coverage_hints(question, retrieved_files)
    if coverage_warnings:
        print("\n⚠️ 檢索覆蓋度提示:")
        for w in coverage_warnings:
            print(f"  - {w}")

    prompt = f"""你是專案程式碼問答助手。請嚴格根據下方提供的程式碼片段用繁體中文回答問題,遵守以下規則:

1. 只能根據片段裡「明確寫出的內容」回答,不可以用程式慣例、常見寫法或任何先驗知識去推測、補全片段中沒有寫的細節(例如:型態、預設值、參數意義、行為邏輯)。
2. 如果某個細節片段中沒有明確標註或說明,必須明確說出「片段中未標註/未提及此細節」,不可以自行猜測或假設。
3. 回答中提到的每一個具體事實(函式名稱、參數、行為),盡量標註是從哪個檔案/行號看到的,方便使用者核對。
4. 如果片段內容不足以完整回答問題,請先回答片段裡確定的部分,再明確指出哪部分找不到依據,不要為了讓回答看起來完整而編造內容。
5. 絕對不要因為問題聽起來「應該」有某個答案,就假設片段裡一定有寫。沒看到就是沒看到。

【問題】
{question}

【相關程式碼片段】
{context_text}
"""
    print("\n查詢本地模型中,請稍候...")
    resp = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    answer = resp['message']['content']
    print(f"\n=== 回答 ===\n{answer}\n")

    # 三層事實查核(取代原本「同一顆模型自己核對自己」的作法):
    #   1. 生成模型(上面已完成)
    #   2. NLI 查核 —— 獨立架構、不同訓練來源的 DeBERTa-v3 模型,逐句判斷語意上有沒有依據
    #   3. 規則式查核 —— 純正則表達式比對數字/日期/識別字/檔名,不靠模型
    context_items = [item for _, item in top_k]
    rule_issues, nli_results = _run_fact_checks(context_items, context_text, retrieved_files, answer)

    report_text = _build_report_text(question, answer, rule_issues, nli_results)

    _flush_stdin()
    save = input("是否要複製「回答＋查核結果」到剪貼簿?(y/n): ").strip().lower()
    if save == "y":
        ok = _copy_to_clipboard(report_text)
        print("已複製到剪貼簿(含查核結果)" if ok else "複製失敗")

def menu():
    while True:
        print("""
========== Agent Project 助理 ==========
1. 顯示資料夾結構(可指定路徑,留空為目前資料夾)
2. 打包程式碼給 Claude(複製到剪貼簿)
3. 用本地模型摘要整理(複製到剪貼簿)
4. 從剪貼簿套用修改(Claude 回覆貼回後用)
5. 用本地模型直接修改(貼 Claude 生成的指令)
6. 用本地模型整理需要修改部份的完整程式碼(貼 Claude 說明後用)
7. 修改完把程式貼回 Claude 檢查
8. 建立/更新 RAG 索引
9. 用 RAG 提問查詢專案程式碼(含三層事實查核)
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
            build_rag_index()
        elif choice == "9":
            query_rag()
        elif choice == "0":
            sys.exit(0)
        else:
            print("無效選項")

if __name__ == "__main__":
    menu()
