#!/usr/bin/env python3
"""
Merge results from split PDF chunks back into single outputs
"""

import os
import sys
import json
import yaml
import re
import shutil
from pathlib import Path
from collections import defaultdict

def adjust_md_page_numbers(content: str, offset: int) -> str:
    """Adjust page numbers in markdown content"""
    # Pattern 1: [Page X]
    def replace_page_bracket(match):
        page_num = int(match.group(1))
        new_page = page_num + offset
        return f"[Page {new_page}]"

    # Pattern 2: ## Page X (section headers)
    def replace_page_header(match):
        page_num = int(match.group(1))
        new_page = page_num + offset
        return f"## Page {new_page}"

    content = re.sub(r'\[Page (\d+)\]', replace_page_bracket, content)
    content = re.sub(r'^## Page (\d+)$', replace_page_header, content, flags=re.MULTILINE)
    return content

def adjust_md_image_references(content: str, offset: int) -> str:
    """Adjust image references in markdown to point to renumbered images"""
    # Pattern: ![...](images/page_XXX.png)
    def replace_image_ref(match):
        alt_text = match.group(1)
        page_num = int(match.group(2))
        new_page_num = page_num + offset
        return f"![{alt_text}](images/page_{new_page_num:03d}.png)"

    content = re.sub(r'!\[(.*?)\]\(images/page_(\d+)\.png\)', replace_image_ref, content)
    return content

def adjust_md_yaml_references(content: str, offset: int) -> str:
    """Adjust YAML file references in markdown"""
    # Pattern: yaml_metadata/XXX_pageN.yaml
    def replace_yaml_ref(match):
        prefix = match.group(1)  # yaml_metadata/Figure_1
        page_num = int(match.group(2))
        new_page_num = page_num + offset
        return f"{prefix}_page{new_page_num}.yaml"

    content = re.sub(r'(yaml_metadata/[^_]+(?:_[^_]+)*)_page(\d+)\.yaml', replace_yaml_ref, content)
    return content

def merge_chunks_for_pdf(original_pdf_name, chunks, results_dir, output_dir):
    """Merge all chunks belonging to one original PDF"""

    # Sort chunks by chunk index
    chunks = sorted(chunks, key=lambda x: x['chunk_idx'])

    print(f"\n  📄 Merging: {original_pdf_name}")
    print(f"      {len(chunks)} chunks to merge")

    # Prepare output directory
    output_pdf_dir = Path(output_dir) / Path(original_pdf_name).stem
    output_pdf_dir.mkdir(parents=True, exist_ok=True)

    # Merge Markdown files
    md_content = []
    total_figures = 0

    # Merge YAML metadata
    all_yaml_metadata = []
    yaml_metadata_dir = output_pdf_dir / "yaml_metadata"
    yaml_metadata_dir.mkdir(exist_ok=True)

    # Merge images
    images_dir = output_pdf_dir / "images"
    images_dir.mkdir(exist_ok=True)

    page_offset = 0

    # Count successful/failed chunks
    chunks_success = 0
    chunks_missing_dir = 0
    chunks_missing_md = 0

    for chunk in chunks:
        chunk_name = Path(chunk['path']).stem
        chunk_result_dir = Path(results_dir) / chunk_name

        if not chunk_result_dir.exists():
            print(f"      ❌ Chunk {chunk['chunk_idx']}/{chunk['total_chunks']}: Directory not found - {chunk_name}")
            chunks_missing_dir += 1
            continue

        # Read chunk MD file
        chunk_md_files = list(chunk_result_dir.glob("*.md"))
        if not chunk_md_files:
            print(f"      ❌ Chunk {chunk['chunk_idx']}/{chunk['total_chunks']}: MD file missing - {chunk_name}")
            chunks_missing_md += 1
            continue

        chunk_md = chunk_md_files[0]
        with open(chunk_md, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if MD file has actual content (not just metadata)
        if len(content.strip()) < 100:
            print(f"      ⚠️  Chunk {chunk['chunk_idx']}/{chunk['total_chunks']}: MD file has too little content ({len(content)} chars) - {chunk_name}")

        # Adjust page numbers in markdown
        if page_offset > 0:
            content = adjust_md_page_numbers(content, page_offset)
            content = adjust_md_image_references(content, page_offset)
            content = adjust_md_yaml_references(content, page_offset)
        md_content.append(content)
        chunks_success += 1

        # Copy YAML files with adjusted page numbers
        chunk_yaml_dir = chunk_result_dir / "yaml_metadata"
        if chunk_yaml_dir.exists():
            for yaml_file in chunk_yaml_dir.glob("*.yaml"):
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)

                # Adjust page number in chart_identification
                new_page_num = None
                if 'chart_identification' in yaml_data and 'page_number' in yaml_data['chart_identification']:
                    yaml_data['chart_identification']['page_number'] += page_offset
                    new_page_num = yaml_data['chart_identification']['page_number']

                # Adjust page number in metadata (also needs adjustment!)
                if 'metadata' in yaml_data and 'page_number' in yaml_data['metadata']:
                    yaml_data['metadata']['page_number'] += page_offset
                    if new_page_num is None:
                        new_page_num = yaml_data['metadata']['page_number']

                # Generate clean filename (avoid duplicate _page_page format)
                # Extract figure name from original filename, remove old page suffix
                stem = yaml_file.stem
                # Remove old _pageXX suffix
                stem_clean = re.sub(r'_page\d+$', '', stem)
                new_yaml_name = f"{stem_clean}_page{new_page_num}.yaml"
                output_yaml_path = yaml_metadata_dir / new_yaml_name

                with open(output_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

                all_yaml_metadata.append(yaml_data)
                total_figures += 1

        # Copy images with adjusted names
        chunk_images_dir = chunk_result_dir / "images"
        if chunk_images_dir.exists():
            for img_file in chunk_images_dir.glob("*.png"):
                # Extract page number from filename (e.g., page_001.png)
                match = re.search(r'page_(\d+)', img_file.name)
                if match:
                    page_num = int(match.group(1))
                    new_page_num = page_num + page_offset
                    new_img_name = f"page_{new_page_num:03d}.png"

                    # Copy image
                    shutil.copy2(img_file, images_dir / new_img_name)

        page_offset += chunk['pages']
        print(f"      ✅ Chunk {chunk['chunk_idx']}/{chunk['total_chunks']}: {chunk['pages']} pages, page offset now {page_offset}")

    # Summary statistics
    chunks_failed = chunks_missing_dir + chunks_missing_md

    # Check if there's enough content
    if chunks_success == 0:
        print(f"      ❌ Merge failed: No valid chunk content!")
        print(f"         - Directories missing: {chunks_missing_dir}")
        print(f"         - MD files missing: {chunks_missing_md}")
        # Don't create empty files, return failure status directly
        return {
            'original_pdf': original_pdf_name,
            'chunks_merged': 0,
            'chunks_failed': chunks_failed,
            'total_figures': 0,
            'total_pages': 0,
            'success': False
        }

    # Write merged markdown
    merged_md_path = output_pdf_dir / f"{Path(original_pdf_name).stem}.md"
    with open(merged_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {Path(original_pdf_name).stem}\n\n")
        f.write(f"*Merged from {chunks_success}/{len(chunks)} chunks*\n\n")
        f.write("---\n\n")
        f.write("\n\n".join(md_content))

    # Write combined YAML (use _all_figures.yaml suffix for consistency with docmind_converter.py)
    combined_yaml_path = output_pdf_dir / f"{Path(original_pdf_name).stem}_all_figures.yaml"
    with open(combined_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_yaml_metadata, f, allow_unicode=True, sort_keys=False)

    # Display merge results
    if chunks_failed > 0:
        print(f"      ⚠️  Merged: {chunks_success}/{len(chunks)} chunks, {total_figures} figures ({chunks_failed} chunks failed)")
    else:
        print(f"      ✅ Merged: {chunks_success}/{len(chunks)} chunks, {total_figures} figures, MD file created")

    return {
        'original_pdf': original_pdf_name,
        'chunks_merged': chunks_success,
        'chunks_failed': chunks_failed,
        'total_figures': total_figures,
        'total_pages': page_offset,
        'success': True
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge chunk results")
    parser.add_argument("--mapping-file", required=True, help="Split mapping JSON file")
    parser.add_argument("--results-dir", required=True, help="Directory containing chunk results")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged results")

    args = parser.parse_args()

    # Load mapping
    with open(args.mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    chunks_info = mapping_data.get('chunks', [])

    # Group chunks by original PDF
    chunks_by_pdf = defaultdict(list)
    for chunk in chunks_info:
        original_pdf = chunk['original_pdf']
        chunks_by_pdf[original_pdf].append(chunk)

    print("="*80)
    print("🔗 Merging Split PDF Results")
    print("="*80)
    print(f"Total original PDFs: {len(chunks_by_pdf)}")
    print(f"Total chunks: {len(chunks_info)}")
    print()

    # Merge each PDF's chunks
    merge_stats = []
    for original_pdf, chunks in chunks_by_pdf.items():
        try:
            result = merge_chunks_for_pdf(original_pdf, chunks, args.results_dir, args.output_dir)
            merge_stats.append(result)
        except Exception as e:
            print(f"  ❌ Error merging {original_pdf}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print()
    print("="*80)

    # Count success/failure
    success_count = sum(1 for s in merge_stats if s.get('success', True))
    failed_count = len(merge_stats) - success_count
    total_chunks_merged = sum(s.get('chunks_merged', 0) for s in merge_stats)
    total_chunks_failed = sum(s.get('chunks_failed', 0) for s in merge_stats)
    total_figures = sum(s['total_figures'] for s in merge_stats)
    total_pages = sum(s['total_pages'] for s in merge_stats)

    if failed_count > 0 or total_chunks_failed > 0:
        print("⚠️  Merging Complete (with issues)")
        print("="*80)
        print(f"PDFs: {success_count} succeeded, {failed_count} failed")
        print(f"Chunks: {total_chunks_merged} succeeded, {total_chunks_failed} failed")
        print(f"Total figures: {total_figures}")
        print(f"Total pages: {total_pages}")
        print()
        print("❌ Failed PDFs:")
        for s in merge_stats:
            if not s.get('success', True) or s.get('chunks_failed', 0) > 0:
                print(f"   - {s['original_pdf']}: {s.get('chunks_merged', 0)} chunks succeeded, {s.get('chunks_failed', 0)} chunks failed")
    else:
        print("✅ Merging Complete")
        print("="*80)
        print(f"PDFs merged: {len(merge_stats)}")
        print(f"Total chunks: {total_chunks_merged}")
        print(f"Total figures: {total_figures}")
        print(f"Total pages: {total_pages}")
    print()

if __name__ == "__main__":
    sys.exit(main())
