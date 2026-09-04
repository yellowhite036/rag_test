#!/usr/bin/env python3
"""
用 Docker 指令一次匯出全部 8 張表到本機 files/ 資料夾
"""

import os
import subprocess
import sys

TABLES = [
    "bom_table",
    "inventory_transactions",
    "materials",
    "molds",
    "products",
    "system_logs",
    "users",
    "work_orders",
]

CONTAINER = "practice_project2-postgres-1"
OUTPUT_DIR = "files"


def run(cmd: list[str]):
    """執行指令，失敗就顯示錯誤"""
    print("→", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ 失敗：", result.stderr.strip())
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== 開始在容器內匯出 ===")
    for t in TABLES:
        run([
            "docker", "exec", CONTAINER,
            "psql", "-U", "postgres", "-d", "practice_project2",
            "-c", f"\\copy {t} TO '/tmp/{t}.csv' WITH CSV HEADER"
        ])

    print("\n=== 複製檔案到本機 ===")
    for t in TABLES:
        run([
            "docker", "cp",
            f"{CONTAINER}:/tmp/{t}.csv",
            os.path.join(OUTPUT_DIR, f"{t}.csv")
        ])

    print(f"\n✅ 全部完成！檔案在 {OUTPUT_DIR}/ 資料夾")


if __name__ == "__main__":
    main()