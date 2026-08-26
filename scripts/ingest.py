#!/usr/bin/env python3
"""
OmniRAG — CLI 导入脚本
用法：
  python scripts/ingest.py --file path/to/file.pdf
  python scripts/ingest.py --dir path/to/folder/
  python scripts/ingest.py --text "一些原始文本" --label "my-doc"
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from app.rag.ingestion import extract_from_text, ingest_directory, ingest_documents, ingest_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OmniRAG 文档导入")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="单个文件的路径")
    group.add_argument("--dir", help="文件目录的路径")
    group.add_argument("--text", help="要导入的原始文本")

    parser.add_argument("--label", default="cli-input", help="原始文本的来源标签")
    parser.add_argument("--extensions", nargs="+", default=[".pdf", ".docx", ".txt", ".md"])

    args = parser.parse_args()

    if args.file:
        logger.info("正在导入文件: %s", args.file)
        count = ingest_file(args.file)
        print(f"✅ 已从 {args.file} 导入 {count} 个分块")

    elif args.dir:
        logger.info("正在导入目录: %s", args.dir)
        count = ingest_directory(args.dir, extensions=args.extensions)
        print(f"✅ 已从 {args.dir} 导入共 {count} 个分块")

    elif args.text:
        docs = extract_from_text(args.text, source=args.label)
        count = ingest_documents(docs)
        print(f"✅ 已从原始文本导入 {count} 个分块")


if __name__ == "__main__":
    main()
