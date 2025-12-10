#!/usr/bin/env python3
"""
Content Filter Recovery - Recover blocked pages using OpenAI GPT-4o-mini

Usage:
    python3 scripts/final_data_validation.py --dry-run           # Preview
    python3 scripts/final_data_validation.py --output-dir output/chunks  # Execute
    python3 scripts/final_data_validation.py --concurrency 15    # 15 concurrent
    python3 scripts/final_data_validation.py --final-delivery final-delivery  # Also update final-delivery

Requires OPENAI_API_KEY environment variable (set in .env file)
"""

import os
import sys
import json
import yaml
import glob
import base64
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not installed, please run: pip install openai")

# Configuration
CHUNKS_DIR = "output/chunks"
FINAL_DELIVERY_DIR = "final-delivery"
REPORT_FILE = "reports/moderation_blocked_pages.json"

# Default concurrency
DEFAULT_CONCURRENCY = 10

# Prompt
EXTRACTION_PROMPT = """请将这张图片中的内容转换为 Markdown 格式。

要求：
1. 保持原有的段落结构和换行
2. 正确识别并格式化标题层级 (# ## ### 等)
3. 表格使用 Markdown 表格语法
4. 数学公式使用 LaTeX 语法 ($..$ 或 $$..$$)
5. 保持原文语言（不要翻译）
6. 不要添加任何解释，只输出转换后的内容

直接输出 Markdown 内容，不要用代码块包裹。"""


@dataclass
class ProcessResult:
    """Processing result"""
    chunk: str
    page: int
    success: bool
    content: str = ""
    error: str = ""
    tokens: int = 0
    duration: float = 0.0


@dataclass
class ProcessStats:
    """Processing statistics"""
    total: int = 0
    success: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_duration: float = 0.0
    errors: List[Dict] = field(default_factory=list)


def load_openai_key() -> str:
    """Load OpenAI API Key from environment variable"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        raise ValueError("OPENAI_API_KEY not set, please configure in .env file")
    return api_key


def load_blocked_pages(chunks_dir: str = CHUNKS_DIR) -> List[Dict]:
    """Load list of blocked pages"""

    # Prefer loading from report file
    report_path = Path(__file__).parent.parent / REPORT_FILE
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            pages = json.load(f)
            # Ensure chunk_dir field exists
            for p in pages:
                if 'chunk_dir' not in p:
                    p['chunk_dir'] = str(Path(__file__).parent.parent / CHUNKS_DIR / p['chunk'])
            return pages

    # Scan from validation.yaml
    blocked = []
    chunks_path = Path(__file__).parent.parent / chunks_dir

    for vfile in glob.glob(str(chunks_path / "*/*.validation.yaml")):
        try:
            with open(vfile, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data or 'failed_pages_detail' not in data:
                continue

            chunk_dir = os.path.dirname(vfile)
            chunk_name = os.path.basename(chunk_dir)

            for item in data.get('failed_pages_detail', []):
                if isinstance(item, dict):
                    error = item.get('error', '')
                    if 'DataInspectionFailed' in str(error) and 'inappropriate' in str(error):
                        page = item.get('page')
                        png_path = os.path.join(chunk_dir, "images", f"page_{page:03d}.png")
                        if os.path.exists(png_path):
                            blocked.append({
                                'chunk': chunk_name,
                                'page': page,
                                'png_path': png_path,
                                'chunk_dir': chunk_dir
                            })
        except Exception as e:
            print(f"Warning: Failed to parse {vfile}: {e}")

    return blocked


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def process_page_async(
    page_info: Dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini"
) -> ProcessResult:
    """Process a single page asynchronously"""

    chunk = page_info['chunk']
    page_num = page_info['page']
    png_path = page_info.get('png_path')

    if not png_path:
        png_path = os.path.join(page_info['chunk_dir'], "images", f"page_{page_num:03d}.png")

    result = ProcessResult(chunk=chunk, page=page_num, success=False)

    if not os.path.exists(png_path):
        result.error = f"PNG not found: {png_path}"
        return result

    async with semaphore:
        start_time = time.time()

        try:
            # Encode image
            base64_image = encode_image(png_path)

            # Call GPT-4o-mini
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )

            result.content = response.choices[0].message.content
            result.tokens = response.usage.total_tokens
            result.success = True

        except Exception as e:
            result.error = str(e)

        result.duration = time.time() - start_time
        return result


def update_markdown_file(chunk_dir: str, page_num: int, content: str) -> bool:
    """Insert recovered content into Markdown file"""
    import re

    # Find the corresponding .md file
    md_files = glob.glob(os.path.join(chunk_dir, "*.md"))
    if not md_files:
        return False

    md_file = md_files[0]

    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            original = f.read()

        # Backup original file
        backup_path = md_file + ".bak"
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)

        # Find and replace page content
        # Format: ## Page X\n[content until next ## Page or end]
        pattern = rf'(## Page {page_num}\n).*?(?=\n## Page |\Z)'
        replacement = f"## Page {page_num}\n\n<!-- Recovered by GPT-4o-mini -->\n{content}\n"

        new_content, count = re.subn(pattern, replacement, original, flags=re.DOTALL)

        if count == 0:
            # Page marker not found, append to end
            new_content = original + f"\n\n## Page {page_num}\n\n<!-- Recovered by GPT-4o-mini -->\n{content}\n"

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"    Failed to update file: {e}")
        return False


def update_final_delivery(chunk_name: str, page_num: int, content: str, final_delivery_dir: str) -> bool:
    """Sync update the corresponding MD file in final-delivery directory"""
    import re

    final_dir = Path(final_delivery_dir)
    if not final_dir.exists():
        return False

    # Extract original PDF name from chunk_name
    # e.g.: "My_Document_part1of3" -> "My_Document" or "My-Document"
    base_name = re.sub(r'_part\d+of\d+$', '', chunk_name)

    # Normalize name for matching
    def normalize(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s.lower())

    norm_base = normalize(base_name)

    # Find matching MD file
    matched_md = None
    for md_file in final_dir.glob('*.md'):
        if md_file.name.startswith('QUALITY_REPORT'):
            continue
        if normalize(md_file.stem).startswith(norm_base[:20]) or norm_base.startswith(normalize(md_file.stem)[:20]):
            matched_md = md_file
            break

    if not matched_md:
        # Try looser matching
        for md_file in final_dir.glob('*.md'):
            if md_file.name.startswith('QUALITY_REPORT'):
                continue
            # Check first 15 characters
            if normalize(md_file.stem)[:15] == norm_base[:15]:
                matched_md = md_file
                break

    if not matched_md:
        return False

    try:
        with open(matched_md, 'r', encoding='utf-8') as f:
            original = f.read()

        # Find and replace page content
        pattern = rf'(## Page {page_num}\n).*?(?=\n## Page |\Z)'
        replacement = f"## Page {page_num}\n\n<!-- Recovered by GPT-4o-mini -->\n{content}\n"

        new_content, count = re.subn(pattern, replacement, original, flags=re.DOTALL)

        if count > 0:
            with open(matched_md, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True

    except Exception as e:
        pass

    return False


def update_validation_file(chunk_dir: str, page_num: int, success: bool):
    """Update validation.yaml to mark page as recovered"""
    validation_file = glob.glob(os.path.join(chunk_dir, "*.validation.yaml"))
    if not validation_file:
        return

    try:
        with open(validation_file[0], 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Update failed_pages_detail
        if 'failed_pages_detail' in data:
            for item in data['failed_pages_detail']:
                if item.get('page') == page_num:
                    if success:
                        item['recovered'] = True
                        item['recovered_by'] = 'gpt-4o-mini'
                        item['recovered_at'] = datetime.now().isoformat()
                    break

        # Add recovery info
        if 'recovery_info' not in data:
            data['recovery_info'] = {
                'recovery_date': datetime.now().isoformat(),
                'pages_recovered': 0,
                'recovery_method': 'gpt-4o-mini'
            }

        if success:
            data['recovery_info']['pages_recovered'] = data['recovery_info'].get('pages_recovered', 0) + 1

        with open(validation_file[0], 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    except Exception as e:
        pass


async def process_all_pages(
    pages: List[Dict],
    api_key: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str = "gpt-4o-mini",
    final_delivery_dir: str = None
) -> ProcessStats:
    """Process all pages in parallel"""

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    stats = ProcessStats(total=len(pages))

    print(f"\nStarting parallel processing of {len(pages)} pages (concurrency: {concurrency})")
    if final_delivery_dir:
        print(f"Also updating final-delivery: {final_delivery_dir}")
    print("=" * 70)

    # Create all tasks
    tasks = [
        process_page_async(page, client, semaphore, model)
        for page in pages
    ]

    # Progress display
    completed = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1

        # Status symbol
        if result.success:
            status = "✓"
            stats.success += 1
            stats.total_tokens += result.tokens

            # Update files
            chunk_dir = None
            for p in pages:
                if p['chunk'] == result.chunk and p['page'] == result.page:
                    chunk_dir = p.get('chunk_dir')
                    break

            if chunk_dir:
                if update_markdown_file(chunk_dir, result.page, result.content):
                    update_validation_file(chunk_dir, result.page, True)
                    # Sync update final-delivery
                    if final_delivery_dir:
                        if update_final_delivery(result.chunk, result.page, result.content, final_delivery_dir):
                            status = "✓+"  # Also updated final-delivery
                        else:
                            status = "✓"  # Only updated chunks
                else:
                    status = "⚠"  # File update failed
        else:
            status = "✗"
            stats.failed += 1
            stats.errors.append({
                'chunk': result.chunk,
                'page': result.page,
                'error': result.error
            })

        stats.total_duration += result.duration

        # Progress output
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(pages) - completed) / rate if rate > 0 else 0

        print(f"  [{completed}/{len(pages)}] {status} {result.chunk} Page {result.page} "
              f"({result.duration:.1f}s, {result.tokens} tokens) "
              f"[{rate:.1f} pages/sec, ETA: {eta:.0f}s]")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Content Filter Recovery - Recover blocked pages using OpenAI GPT-4o-mini',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/final_data_validation.py --dry-run                    # Preview
    python3 scripts/final_data_validation.py --output-dir output/chunks   # Execute
    python3 scripts/final_data_validation.py --concurrency 15             # 15 concurrent
    python3 scripts/final_data_validation.py --final-delivery final-delivery  # Also update final-delivery

Requires OPENAI_API_KEY environment variable (set in .env file)
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview only, no actual processing')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Concurrency (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of pages to process (0=all)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model (default: gpt-4o-mini)')
    parser.add_argument('--chunks-dir', type=str, default=CHUNKS_DIR,
                        help=f'Chunks directory (default: {CHUNKS_DIR})')
    parser.add_argument('--final-delivery', '-f', type=str, default=None,
                        help=f'Also update final-delivery directory (default: no update)')

    args = parser.parse_args()

    if not OPENAI_AVAILABLE:
        print("Error: Please install openai library first")
        print("  pip install openai")
        sys.exit(1)

    print("=" * 70)
    print("DocMind Content Filter Recovery Tool (Parallel)")
    print("=" * 70)

    # Load API Key
    try:
        api_key = load_openai_key()
        print(f"✓ API Key loaded: [configured]")
    except Exception as e:
        print(f"✗ Failed to load API Key: {e}")
        sys.exit(1)

    # Load blocked pages
    blocked = load_blocked_pages(args.chunks_dir)
    print(f"✓ Found {len(blocked)} blocked pages")

    if args.limit > 0:
        blocked = blocked[:args.limit]
        print(f"  Limited to first {args.limit} pages")

    if not blocked:
        print("\nNo pages to process")
        return

    # Cost estimation
    estimated_tokens = len(blocked) * 40000
    estimated_cost_usd = estimated_tokens * 0.15 / 1000000  # $0.15/1M input tokens
    estimated_cost_cny = estimated_cost_usd * 7.2

    print(f"\nEstimates:")
    print(f"  - Tokens: ~{estimated_tokens:,}")
    print(f"  - Cost: ${estimated_cost_usd:.4f} (~¥{estimated_cost_cny:.2f})")
    print(f"  - Concurrency: {args.concurrency}")
    print(f"  - Est. time: ~{len(blocked) / args.concurrency * 3:.0f} seconds")

    if args.dry_run:
        print("\n[Preview mode] Pages to process:")
        for i, page in enumerate(blocked[:20]):
            png_exists = "✓" if os.path.exists(page.get('png_path', '')) else "✗"
            print(f"  {i+1}. {page['chunk']} - Page {page['page']} [PNG: {png_exists}]")
        if len(blocked) > 20:
            print(f"  ... and {len(blocked) - 20} more pages")
        return

    # Confirm execution
    print(f"\nAbout to process {len(blocked)} pages, estimated cost ¥{estimated_cost_cny:.2f}")
    confirm = input("Continue? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return

    # Check final-delivery directory
    final_delivery_dir = None
    if args.final_delivery:
        final_delivery_path = Path(__file__).parent.parent / args.final_delivery
        if final_delivery_path.exists():
            final_delivery_dir = str(final_delivery_path)
            print(f"✓ Will also update final-delivery: {final_delivery_dir}")
        else:
            print(f"⚠ final-delivery directory not found: {final_delivery_path}")

    # Execute processing
    stats = asyncio.run(process_all_pages(
        blocked,
        api_key,
        concurrency=args.concurrency,
        model=args.model,
        final_delivery_dir=final_delivery_dir
    ))

    # Output statistics
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"  Total pages: {stats.total}")
    print(f"  Success: {stats.success}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total Tokens: {stats.total_tokens:,}")
    print(f"  Total duration: {stats.total_duration:.1f}s")
    print(f"  Average speed: {stats.total / stats.total_duration:.1f} pages/sec" if stats.total_duration > 0 else "")

    if stats.errors:
        print(f"\nFailed pages:")
        for err in stats.errors[:10]:
            print(f"  - {err['chunk']} Page {err['page']}: {err['error'][:50]}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more errors")

    # Save report
    report_path = Path(__file__).parent.parent / "reports" / "recovery_report.json"
    report_path.parent.mkdir(exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'concurrency': args.concurrency,
        'stats': {
            'total': stats.total,
            'success': stats.success,
            'failed': stats.failed,
            'total_tokens': stats.total_tokens,
            'total_duration': stats.total_duration
        },
        'errors': stats.errors
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
