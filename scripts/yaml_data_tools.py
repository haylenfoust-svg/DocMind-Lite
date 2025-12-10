#!/usr/bin/env python3
"""
YAML Data Tools - DocMind Lite

Features:
  - fill:    Fill YAML table data from Markdown
  - sync:    Sync Table*.yaml to _all_figures.yaml
  - update:  Update final-delivery YAML
  - verify:  Verify YAML and MD data consistency
  - preview: Generate Markdown table preview from YAML

Usage:
    python scripts/yaml_data_tools.py <command> [options]

Examples:
    python scripts/yaml_data_tools.py fill --output-dir output/chunks
    python scripts/yaml_data_tools.py sync --output-dir output/chunks
    python scripts/yaml_data_tools.py update --chunks-dir output/chunks --final-delivery final-delivery
    python scripts/yaml_data_tools.py verify --output-dir output/chunks
    python scripts/yaml_data_tools.py preview --yaml-file path/to/file.yaml

Created: 2025-12-08
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
from datetime import datetime


# ============================================================================
# Common Utility Functions
# ============================================================================

def load_yaml(yaml_path: str) -> Optional[Dict]:
    """Load YAML file"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None


def save_yaml(yaml_path: str, data: Any) -> bool:
    """Save YAML file"""
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving {yaml_path}: {e}")
        return False


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings (0-1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def normalize_value(val) -> str:
    """Normalize value for comparison"""
    if val is None:
        return ''
    s = str(val).strip()
    s = s.replace(',', '')
    s = s.replace('–', '-').replace('—', '-')
    return s.lower()


def is_numeric(val: str) -> bool:
    """Check if value is numeric"""
    if not val:
        return False
    clean = re.sub(r'[£$€¥%,\s\-–—?]', '', val)
    if not clean:
        return False
    try:
        float(clean)
        return True
    except ValueError:
        return False


def try_convert_number(value: str) -> Any:
    """Try to convert to numeric value"""
    if not value:
        return value
    clean = re.sub(r'[£$€¥,\s]', '', value)
    if '%' in value:
        clean = clean.replace('%', '')
        try:
            return float(clean)
        except ValueError:
            return value
    try:
        if '.' in clean:
            return float(clean)
        else:
            return int(clean)
    except ValueError:
        return value


def extract_keywords(text: str) -> set:
    """Extract keywords"""
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.lower().split()
    return {w for w in words if len(w) > 2}


def keyword_overlap(a: str, b: str) -> float:
    """Calculate keyword overlap ratio"""
    kw_a = extract_keywords(a)
    kw_b = extract_keywords(b)
    if not kw_a or not kw_b:
        return 0.0
    intersection = kw_a & kw_b
    union = kw_a | kw_b
    return len(intersection) / len(union) if union else 0.0


def count_yaml_data_rows(data_series: Any) -> int:
    """Count the number of data rows in YAML data_series"""
    if not data_series or not isinstance(data_series, list):
        return 0
    if len(data_series) == 0:
        return 0
    first_item = data_series[0]
    if not isinstance(first_item, dict):
        return len(data_series)
    if 'data_points' in first_item and isinstance(first_item['data_points'], list):
        return sum(len(item.get('data_points', [])) for item in data_series)
    else:
        return len(data_series)


# ============================================================================
# FILL: Fill YAML table data from Markdown
# ============================================================================

def parse_table_row(line: str) -> List[str]:
    """Parse table row"""
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    cells = [c.strip() for c in line.split('|')]
    return cells


def parse_table_content(table_content: str) -> Optional[Dict[str, Any]]:
    """Parse table content"""
    lines = [l.strip() for l in table_content.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return None

    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('|'):
            header_idx = i
            break
    if header_idx < 0:
        return None

    header_line = lines[header_idx]
    headers = parse_table_row(header_line)
    if not headers:
        return None

    sep_idx = header_idx + 1
    if sep_idx >= len(lines):
        return None

    sep_line = lines[sep_idx]
    if not re.match(r'\|[\s\-:|]+\|', sep_line):
        sep_idx = header_idx

    data_start_idx = sep_idx + 1

    rows = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i]
        if not line.startswith('|'):
            continue
        cells = parse_table_row(line)
        if cells:
            row_data = {'x': cells[0]}
            for j, c in enumerate(cells[1:], 1):
                key = headers[j] if j < len(headers) else f"col_{j}"
                row_data[key] = try_convert_number(c)
            if len(row_data) > 1:
                rows.append(row_data)

    if not rows:
        return None
    return {'headers': headers, 'rows': rows}


def find_tables_in_markdown(md_content: str, page_num: int = 0) -> List[Dict[str, Any]]:
    """Extract tables from Markdown"""
    tables = []

    if page_num > 0:
        pages_to_search = [page_num - 1, page_num, page_num + 1]
    else:
        pages_to_search = None

    def extract_from_section(section_content: str, source_page: int):
        section_tables = []
        pattern = r'###\s*(Table[^\n]*)\n\n?((?:\|[^\n]+\n)+)'
        for match in re.finditer(pattern, section_content, re.MULTILINE):
            raw_title = match.group(1).strip()
            table_content = match.group(2).strip()
            if '|' in raw_title:
                parts = raw_title.split('|')
                title = parts[0].strip().rstrip(':').strip()
                header_part = '|' + '|'.join(parts[1:])
                if not header_part.endswith('\n'):
                    header_part += '\n'
                table_content = header_part + table_content
            else:
                title = raw_title
            parsed = parse_table_content(table_content)
            if parsed:
                section_tables.append({
                    'title': title,
                    'content': table_content,
                    'page': source_page,
                    **parsed
                })
        return section_tables

    if pages_to_search:
        for search_page in pages_to_search:
            if search_page < 1:
                continue
            page_pattern = rf'## Page {search_page}\b(.*?)(?=## Page \d+|$)'
            page_match = re.search(page_pattern, md_content, re.DOTALL)
            if page_match:
                tables.extend(extract_from_section(page_match.group(1), search_page))
    else:
        tables.extend(extract_from_section(md_content, 0))

    return tables


def find_best_matching_table(yaml_data: Dict, tables: List[Dict], page_num: int) -> Optional[Dict]:
    """Find the best matching table for YAML"""
    chart_title = yaml_data.get('chart_identification', {}).get('chart_title', '')
    key_elements = yaml_data.get('visual_content', {}).get('key_elements', [])
    figure_number = yaml_data.get('chart_identification', {}).get('figure_number', '')

    if not tables:
        return None

    best_match = None
    best_score = 0.0
    title_is_empty = not chart_title or chart_title.strip() == ''

    for table in tables:
        score = 0.0
        if not title_is_empty:
            if table['title']:
                score += similarity(chart_title, table['title']) * 0.4
                score += keyword_overlap(chart_title, table['title']) * 0.3
            if key_elements and table.get('headers'):
                header_str = ' '.join(table['headers'])
                elements_str = ' '.join(key_elements)
                score += keyword_overlap(header_str, elements_str) * 0.3
        else:
            if key_elements and table.get('headers'):
                header_str = ' '.join(table['headers'])
                elements_str = ' '.join(key_elements)
                score += keyword_overlap(header_str, elements_str) * 0.6
            if figure_number and table['title']:
                if figure_number.lower() in table['title'].lower():
                    score += 0.4

        if score > best_score:
            best_score = score
            best_match = table

    threshold = 0.15 if title_is_empty else 0.2
    if best_score < threshold:
        if len(tables) == 1:
            return tables[0]
        return None
    return best_match


def cmd_fill(args):
    """Execute fill command: Fill YAML table data from MD"""
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory does not exist: {output_dir}")
        return 1

    print("=" * 70)
    print("YAML Table Data Fill Tool")
    print("=" * 70)
    print(f"Directory: {output_dir}")
    print(f"Mode: {'Check' if args.dry_run else 'Execute'}")
    if args.force:
        print(f"Force overwrite: Yes")
    print()

    stats = {'total': 0, 'filled': 0, 'skipped': 0, 'failed': 0}

    for chunk_dir in sorted(output_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        yaml_dir = chunk_dir / 'yaml_metadata'
        if not yaml_dir.exists():
            continue

        md_files = list(chunk_dir.glob('*.md'))
        if not md_files:
            continue
        md_path = md_files[0]

        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except:
            continue

        for yaml_path in sorted(yaml_dir.glob('Table*.yaml')):
            stats['total'] += 1

            yaml_data = load_yaml(str(yaml_path))
            if not yaml_data:
                stats['failed'] += 1
                continue

            image_type = yaml_data.get('chart_identification', {}).get('image_type', '')
            if image_type not in ['data_table', 'table']:
                stats['skipped'] += 1
                continue

            existing_series = yaml_data.get('data_extraction', {}).get('data_series', [])
            if existing_series and not args.force:
                stats['skipped'] += 1
                continue

            page_num = 0
            page_match = re.search(r'page(\d+)', yaml_path.name)
            if page_match:
                page_num = int(page_match.group(1))

            tables = find_tables_in_markdown(md_content, page_num)
            if not tables:
                stats['failed'] += 1
                continue

            best_table = find_best_matching_table(yaml_data, tables, page_num)
            if not best_table:
                stats['failed'] += 1
                continue

            data_series = best_table.get('rows', [])
            if not data_series:
                stats['failed'] += 1
                continue

            if not args.dry_run:
                if 'data_extraction' not in yaml_data:
                    yaml_data['data_extraction'] = {}
                yaml_data['data_extraction']['data_series'] = data_series
                yaml_data['data_extraction']['has_quantitative_data'] = True
                save_yaml(str(yaml_path), yaml_data)

            stats['filled'] += 1
            print(f"  ✅ [{chunk_dir.name}] {yaml_path.name} → {len(data_series)} rows")

    print()
    print("=" * 70)
    print(f"Total: {stats['total']} | Filled: {stats['filled']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    return 0


# ============================================================================
# SYNC: Sync Table*.yaml to _all_figures.yaml
# ============================================================================

def find_matching_entry(figures: List[Dict], figure_number: str, page_number: int) -> Optional[int]:
    """Find matching entry index in figures list"""
    for i, fig in enumerate(figures):
        chart_id = fig.get('chart_identification', {})
        fig_num = chart_id.get('figure_number', '')
        fig_page = chart_id.get('page_number', 0)
        if fig_num == figure_number and fig_page == page_number:
            return i
    return None


def cmd_sync(args):
    """Execute sync command: Sync Table*.yaml to _all_figures.yaml"""
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory does not exist: {output_dir}")
        return 1

    print("=" * 70)
    print("Sync Table YAML to _all_figures.yaml")
    print("=" * 70)
    print(f"Directory: {output_dir}")
    print()

    stats = {'chunks': 0, 'synced': 0, 'skipped': 0}

    for chunk_dir in sorted(output_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        yaml_dir = chunk_dir / 'yaml_metadata'
        if not yaml_dir.exists():
            continue

        all_figures_files = list(chunk_dir.glob('*_all_figures.yaml'))
        if not all_figures_files:
            continue
        all_figures_path = all_figures_files[0]

        all_figures_data = load_yaml(str(all_figures_path))
        if not all_figures_data:
            continue

        if isinstance(all_figures_data, list):
            figures_list = all_figures_data
            is_wrapped = False
        elif isinstance(all_figures_data, dict):
            figures_list = all_figures_data.get('figures', [])
            is_wrapped = True
        else:
            continue

        stats['chunks'] += 1
        chunk_synced = 0

        for table_yaml_path in sorted(yaml_dir.glob('Table*.yaml')):
            table_data = load_yaml(str(table_yaml_path))
            if not table_data:
                continue

            data_series = table_data.get('data_extraction', {}).get('data_series', [])
            if not data_series:
                stats['skipped'] += 1
                continue

            chart_id = table_data.get('chart_identification', {})
            figure_number = chart_id.get('figure_number', '')
            page_number = chart_id.get('page_number', 0)

            idx = find_matching_entry(figures_list, figure_number, page_number)
            if idx is not None:
                if 'data_extraction' not in figures_list[idx]:
                    figures_list[idx]['data_extraction'] = {}
                figures_list[idx]['data_extraction']['data_series'] = data_series
                figures_list[idx]['data_extraction']['has_quantitative_data'] = True
                chunk_synced += 1
                stats['synced'] += 1

        if chunk_synced > 0 and not args.dry_run:
            if is_wrapped:
                all_figures_data['figures'] = figures_list
                save_yaml(str(all_figures_path), all_figures_data)
            else:
                save_yaml(str(all_figures_path), figures_list)
            print(f"  ✅ {chunk_dir.name}: Synced {chunk_synced} tables")

    print()
    print("=" * 70)
    print(f"Chunks: {stats['chunks']} | Synced: {stats['synced']} | Skipped: {stats['skipped']}")
    return 0


# ============================================================================
# UPDATE: Update final-delivery YAML
# ============================================================================

# Manual mapping for special file names (add entries if needed)
MANUAL_MAPPING = {}


def normalize_name(name: str) -> str:
    """Normalize filename for matching"""
    name = re.sub(r'_part\d+of\d+', '', name)
    name = re.sub(r'_chunk\d+', '', name)
    name = name.replace('_', '').replace('-', '').replace(' ', '')
    return name.lower()


def merge_figures_data(existing_figures: List[Dict], new_figures: List[Dict], page_offset: int = 0) -> List[Dict]:
    """Merge figure data, update data_series"""
    existing_map = {}
    for fig in existing_figures:
        chart_id = fig.get('chart_identification', {})
        key = (chart_id.get('figure_number', ''), chart_id.get('page_number', 0))
        existing_map[key] = fig

    for new_fig in new_figures:
        chart_id = new_fig.get('chart_identification', {})
        orig_page = chart_id.get('page_number', 0)
        adjusted_page = orig_page + page_offset
        key = (chart_id.get('figure_number', ''), adjusted_page)

        if key in existing_map:
            new_series = new_fig.get('data_extraction', {}).get('data_series', [])
            if new_series:
                if 'data_extraction' not in existing_map[key]:
                    existing_map[key]['data_extraction'] = {}
                existing_map[key]['data_extraction']['data_series'] = new_series
                existing_map[key]['data_extraction']['has_quantitative_data'] = True

    return list(existing_map.values())


def cmd_update(args):
    """Execute update command: Update final-delivery YAML"""
    chunks_dir = Path(args.chunks_dir)
    final_dir = Path(args.final_delivery)

    if not chunks_dir.exists():
        print(f"Error: Chunks directory does not exist: {chunks_dir}")
        return 1
    if not final_dir.exists():
        print(f"Error: Final-delivery directory does not exist: {final_dir}")
        return 1

    print("=" * 70)
    print("Update Final Delivery YAML")
    print("=" * 70)
    print(f"Chunks: {chunks_dir}")
    print(f"Final:  {final_dir}")
    print()

    chunk_data = {}
    for chunk_dir in chunks_dir.iterdir():
        if not chunk_dir.is_dir():
            continue
        all_figures_files = list(chunk_dir.glob('*_all_figures.yaml'))
        if not all_figures_files:
            continue
        data = load_yaml(str(all_figures_files[0]))
        if data:
            if isinstance(data, dict):
                figures = data.get('figures', [])
            else:
                figures = data
            chunk_data[chunk_dir.name] = figures

    stats = {'updated': 0, 'skipped': 0}

    for final_yaml in sorted(final_dir.glob('*.yaml')):
        if final_yaml.name.startswith('QUALITY') or final_yaml.name.startswith('VALIDATION'):
            continue

        final_data = load_yaml(str(final_yaml))
        if not final_data:
            continue

        if isinstance(final_data, dict):
            final_figures = final_data.get('figures', [])
            is_wrapped = True
        else:
            final_figures = final_data
            is_wrapped = False

        base_name = final_yaml.stem.replace('_all_figures', '')
        norm_base = normalize_name(base_name)

        matched_chunks = []
        for chunk_name, figures in chunk_data.items():
            if normalize_name(chunk_name).startswith(norm_base):
                matched_chunks.append((chunk_name, figures))

        if not matched_chunks:
            for mapped_name, original_name in MANUAL_MAPPING.items():
                if normalize_name(mapped_name) == norm_base:
                    for chunk_name, figures in chunk_data.items():
                        if normalize_name(chunk_name).startswith(normalize_name(original_name)):
                            matched_chunks.append((chunk_name, figures))

        if matched_chunks:
            matched_chunks.sort(key=lambda x: x[0])
            page_offset = 0
            for chunk_name, chunk_figures in matched_chunks:
                final_figures = merge_figures_data(final_figures, chunk_figures, page_offset)
                part_match = re.search(r'_part(\d+)of(\d+)', chunk_name)
                if part_match:
                    pass

            if not args.dry_run:
                if is_wrapped:
                    final_data['figures'] = final_figures
                    save_yaml(str(final_yaml), final_data)
                else:
                    save_yaml(str(final_yaml), final_figures)

            stats['updated'] += 1
            print(f"  ✅ {final_yaml.name} ← {len(matched_chunks)} chunks")
        else:
            stats['skipped'] += 1

    print()
    print("=" * 70)
    print(f"Updated: {stats['updated']} | Skipped: {stats['skipped']}")
    return 0


# ============================================================================
# VERIFY: Verify YAML and MD data consistency
# ============================================================================

def extract_tables_from_markdown(md_path: str, page_num: int) -> List[Dict]:
    """Extract tables from specified page in Markdown"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return []

    all_tables = []
    pages_to_search = [page_num - 1, page_num, page_num + 1]

    for search_page in pages_to_search:
        if search_page < 1:
            continue
        page_pattern = rf'## Page {search_page}\b(.*?)(?=## Page \d+|$)'
        page_match = re.search(page_pattern, content, re.DOTALL)
        if not page_match:
            continue

        page_content = page_match.group(1)
        table_pattern = r'###\s*(Table[^\n:]*):?\s*([^\|\n][^\n]*)?\n+((?:\|[^\n]+\|\n)+)'

        for match in re.finditer(table_pattern, page_content):
            figure_number = match.group(1).strip()
            table_title = match.group(2).strip() if match.group(2) else figure_number
            table_content = match.group(3)

            lines = table_content.strip().split('\n')
            if len(lines) < 2:
                continue

            header_line = lines[0]
            headers = [h.strip() for h in header_line.split('|') if h.strip()]

            data_rows = []
            for line in lines[2:]:
                if line.strip() and '|' in line:
                    cells = [c.strip() for c in line.split('|') if c.strip() or c == '']
                    cells = [c for c in cells if c]
                    if cells:
                        data_rows.append(cells)

            all_tables.append({
                'figure_number': figure_number,
                'title': table_title,
                'headers': headers,
                'data_rows': data_rows,
                'row_count': len(data_rows),
                'page': search_page
            })

    return all_tables


def cmd_verify(args):
    """Execute verify command: Verify YAML and MD consistency"""
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory does not exist: {output_dir}")
        return 1

    print("=" * 70)
    print("YAML Table Data Verification")
    print("=" * 70)
    print(f"Directory: {output_dir}")
    print()

    yaml_files = []
    for chunk_dir in output_dir.iterdir():
        if chunk_dir.is_dir():
            yaml_dir = chunk_dir / 'yaml_metadata'
            if yaml_dir.exists():
                for yaml_file in yaml_dir.glob('Table*.yaml'):
                    md_file = chunk_dir / f'{chunk_dir.name}.md'
                    if md_file.exists():
                        yaml_files.append((str(yaml_file), str(md_file), chunk_dir.name))

    stats = {'PASS': 0, 'PARTIAL': 0, 'EMPTY': 0, 'FAIL': 0}

    print(f"{'Status':<10} {'YAML rows':<10} {'MD rows':<10} {'File'}")
    print("-" * 70)

    for yaml_path, md_path, chunk_name in sorted(yaml_files):
        yaml_data = load_yaml(yaml_path)
        if not yaml_data:
            stats['FAIL'] += 1
            continue

        data_series = yaml_data.get('data_extraction', {}).get('data_series', [])
        yaml_rows = count_yaml_data_rows(data_series)

        if yaml_rows == 0:
            stats['EMPTY'] += 1
            print(f"{'❌ EMPTY':<10} {yaml_rows:<10} {'-':<10} [{chunk_name}] {Path(yaml_path).name}")
            continue

        page_num = 0
        page_match = re.search(r'page(\d+)', yaml_path)
        if page_match:
            page_num = int(page_match.group(1))

        md_tables = extract_tables_from_markdown(md_path, page_num)
        if not md_tables:
            stats['PARTIAL'] += 1
            print(f"{'⚠️ NO_MD':<10} {yaml_rows:<10} {'?':<10} [{chunk_name}] {Path(yaml_path).name}")
            continue

        best_match = md_tables[0]
        for table in md_tables:
            yaml_fn = yaml_data.get('chart_identification', {}).get('figure_number', '').lower()
            if yaml_fn and yaml_fn in table.get('figure_number', '').lower():
                best_match = table
                break

        md_rows = best_match['row_count']

        if yaml_rows == md_rows:
            stats['PASS'] += 1
            print(f"{'✅ PASS':<10} {yaml_rows:<10} {md_rows:<10} [{chunk_name}] {Path(yaml_path).name}")
        else:
            stats['PARTIAL'] += 1
            print(f"{'⚠️ DIFF':<10} {yaml_rows:<10} {md_rows:<10} [{chunk_name}] {Path(yaml_path).name}")

    print()
    print("=" * 70)
    total = sum(stats.values())
    print(f"Total: {total} | PASS: {stats['PASS']} | PARTIAL: {stats['PARTIAL']} | EMPTY: {stats['EMPTY']} | FAIL: {stats['FAIL']}")
    if total > 0:
        pass_rate = (stats['PASS'] + stats['PARTIAL']) / total * 100
        print(f"Pass rate: {pass_rate:.1f}%")
    return 0


# ============================================================================
# PREVIEW: Generate Markdown table preview from YAML
# ============================================================================

def yaml_to_markdown_table(yaml_data: Dict) -> Optional[str]:
    """Convert YAML data_series to Markdown table"""
    data_series = yaml_data.get('data_extraction', {}).get('data_series', [])
    if not data_series:
        return None

    all_keys = set()
    for row in data_series:
        all_keys.update(row.keys())

    columns = ['x'] if 'x' in all_keys else []
    columns.extend(sorted([k for k in all_keys if k != 'x']))

    header = '| ' + ' | '.join(columns) + ' |'
    separator = '|' + '|'.join(['---' for _ in columns]) + '|'

    rows = []
    for row_data in data_series:
        cells = []
        for col in columns:
            val = row_data.get(col, '')
            if val is None:
                val = ''
            elif isinstance(val, float):
                if val == int(val):
                    val = str(int(val))
                else:
                    val = str(val)
            else:
                val = str(val)
            cells.append(val)
        rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join([header, separator] + rows)


def cmd_preview(args):
    """Execute preview command: Generate MD table preview from YAML"""
    yaml_path = Path(args.yaml_file)

    if not yaml_path.exists():
        print(f"Error: File does not exist: {yaml_path}")
        return 1

    yaml_data = load_yaml(str(yaml_path))
    if not yaml_data:
        print(f"Error: Unable to load YAML file")
        return 1

    chart_title = yaml_data.get('chart_identification', {}).get('chart_title', 'Unknown')
    figure_number = yaml_data.get('chart_identification', {}).get('figure_number', '')

    print("=" * 70)
    print(f"YAML Table Preview: {yaml_path.name}")
    print("=" * 70)
    print(f"Figure: {figure_number}")
    print(f"Title:  {chart_title[:60]}...")
    print()

    table_md = yaml_to_markdown_table(yaml_data)
    if table_md:
        print(table_md)
    else:
        print("(No data_series data)")

    print()
    return 0


# ============================================================================
# Main Program
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='YAML Data Tools - DocMind Lite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fill --output-dir output/chunks
  %(prog)s sync --output-dir output/chunks
  %(prog)s update --chunks-dir output/chunks --final-delivery final-delivery
  %(prog)s verify --output-dir output/chunks
  %(prog)s preview --yaml-file path/to/Table_1_page5.yaml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # fill command
    fill_parser = subparsers.add_parser('fill', help='Fill YAML table data from MD')
    fill_parser.add_argument('--output-dir', '-o', default='output/chunks', help='Output directory')
    fill_parser.add_argument('--dry-run', '-n', action='store_true', help='Check only, no modifications')
    fill_parser.add_argument('--force', '-f', action='store_true', help='Force overwrite existing data')

    # sync command
    sync_parser = subparsers.add_parser('sync', help='Sync Table*.yaml to _all_figures.yaml')
    sync_parser.add_argument('--output-dir', '-o', default='output/chunks', help='Output directory')
    sync_parser.add_argument('--dry-run', '-n', action='store_true', help='Check only, no modifications')

    # update command
    update_parser = subparsers.add_parser('update', help='Update final-delivery YAML')
    update_parser.add_argument('--chunks-dir', '-c', default='output/chunks', help='Chunks directory')
    update_parser.add_argument('--final-delivery', '-f', default='final-delivery', help='Final-delivery directory')
    update_parser.add_argument('--dry-run', '-n', action='store_true', help='Check only, no modifications')

    # verify command
    verify_parser = subparsers.add_parser('verify', help='Verify YAML and MD consistency')
    verify_parser.add_argument('--output-dir', '-o', default='output/chunks', help='Output directory')

    # preview command
    preview_parser = subparsers.add_parser('preview', help='Generate MD table preview from YAML')
    preview_parser.add_argument('--yaml-file', '-y', required=True, help='YAML file path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'fill':
        return cmd_fill(args)
    elif args.command == 'sync':
        return cmd_sync(args)
    elif args.command == 'update':
        return cmd_update(args)
    elif args.command == 'verify':
        return cmd_verify(args)
    elif args.command == 'preview':
        return cmd_preview(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
