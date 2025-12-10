#!/usr/bin/env python3
"""
Smart PDF Splitter - considers BOTH page count AND file size
Supports checkpoint resume: skips already split PDFs
"""

import os
import sys
import json
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter

# Import progress manager
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from progress_manager import ProgressManager, get_progress_manager
    PROGRESS_ENABLED = True
except ImportError:
    PROGRESS_ENABLED = False

def split_pdf(pdf_path, output_dir, max_pages_per_chunk=50, max_size_mb_per_chunk=50):
    """Split a PDF into chunks based on pages AND file size"""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    mb_per_page = file_size_mb / total_pages if total_pages > 0 else 0

    print(f"\n  📄 {pdf_path.name}")
    print(f"      Pages: {total_pages}, Size: {file_size_mb:.1f}M, Density: {mb_per_page:.2f}M/page")

    # Calculate chunks needed based on pages
    chunks_by_pages = (total_pages + max_pages_per_chunk - 1) // max_pages_per_chunk

    # Calculate chunks needed based on file size
    chunks_by_size = int((file_size_mb + max_size_mb_per_chunk - 1) / max_size_mb_per_chunk)

    # Take the more restrictive limit (more chunks = smaller chunks)
    num_chunks = max(chunks_by_pages, chunks_by_size)

    if num_chunks <= 1:
        print(f"      ⏭️  No split needed")
        return []

    # Calculate pages per chunk
    pages_per_chunk = (total_pages + num_chunks - 1) // num_chunks
    size_per_chunk = file_size_mb / num_chunks

    print(f"      ✂️  Splitting into {num_chunks} chunks")
    print(f"         Strategy: {chunks_by_pages} by pages, {chunks_by_size} by size → using {num_chunks}")
    print(f"         Each chunk: ~{pages_per_chunk} pages, ~{size_per_chunk:.1f}M")

    chunks = []
    for chunk_idx in range(num_chunks):
        start_page = chunk_idx * pages_per_chunk
        end_page = min((chunk_idx + 1) * pages_per_chunk, total_pages)

        # Create output filename
        base_name = pdf_path.stem
        chunk_name = f"{base_name}_part{chunk_idx+1}of{num_chunks}.pdf"
        chunk_path = output_dir / chunk_name

        # Create chunk PDF
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        with open(chunk_path, 'wb') as f:
            writer.write(f)

        chunk_size = chunk_path.stat().st_size / (1024 * 1024)
        chunks.append({
            'path': str(chunk_path),
            'chunk_idx': chunk_idx + 1,
            'total_chunks': num_chunks,
            'start_page': start_page + 1,  # 1-indexed
            'end_page': end_page,
            'pages': end_page - start_page,
            'size_mb': round(chunk_size, 1),
            'original_pdf': str(pdf_path)
        })

        print(f"         ✅ Chunk {chunk_idx+1}/{num_chunks}: pages {start_page+1}-{end_page} ({chunk_size:.1f}M)")

    return chunks

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart PDF Splitter")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--output-dir", default="./split_pdfs", help="Output directory for chunks")
    parser.add_argument("--threshold", type=int, default=50, help="Page threshold for splitting")
    parser.add_argument("--chunk-size", type=int, default=50, help="Max pages per chunk")
    parser.add_argument("--max-chunk-size-mb", type=int, default=50, help="Max MB per chunk")
    parser.add_argument("--mapping-file", default="./split_mapping.json", help="Output mapping file")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume, re-split all PDFs")

    args = parser.parse_args()

    resume = not args.no_resume

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mapping_file = Path(args.mapping_file)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    # Initialize progress manager
    progress_manager = None
    if PROGRESS_ENABLED and resume:
        base_dir = Path(__file__).parent.parent
        progress_manager = get_progress_manager(None, base_dir)
        progress_manager.load()

    # Check if split step is already completed
    existing_mapping = None
    if resume and mapping_file.exists():
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
            print(f"📌 Found existing split record: {mapping_file}")
        except:
            existing_mapping = None

    # Check status in progress manager
    if progress_manager and progress_manager.is_step_completed('split'):
        if existing_mapping:
            print(f"✅ Split step already completed, skipping")
            print(f"   PDFs split: {existing_mapping.get('stats', {}).get('split_pdfs', 0)}")
            print(f"   Total chunks: {existing_mapping.get('stats', {}).get('total_chunks', 0)}")
            return 0

    # Find all PDFs
    pdf_files = sorted(input_dir.glob("*.pdf"))

    print("="*80)
    print("✂️  Smart PDF Splitting Tool")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split threshold: {args.threshold} pages")
    print(f"Max pages per chunk: {args.chunk_size}")
    print(f"Max size per chunk: {args.max_chunk_size_mb}M")
    if resume:
        print(f"Checkpoint resume: enabled")
    print()

    # Analyze and split
    all_chunks = []
    stats = {
        'total_pdfs': len(pdf_files),
        'split_pdfs': 0,
        'skipped_pdfs': 0,
        'total_chunks': 0,
        'pdfs': {}
    }

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            total_pages = len(reader.pages)
            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)

            # Check page threshold or file size threshold (split if either exceeds)
            needs_split = total_pages > args.threshold or file_size_mb > args.max_chunk_size_mb

            if needs_split:
                chunks = split_pdf(
                    pdf_file,
                    output_dir,
                    args.chunk_size,
                    args.max_chunk_size_mb
                )

                if chunks:  # Only count if actually split
                    all_chunks.extend(chunks)
                    stats['split_pdfs'] += 1
                    stats['total_chunks'] += len(chunks)
                    stats['pdfs'][pdf_file.name] = {
                        'pages': total_pages,
                        'size_mb': round(file_size_mb, 1),
                        'chunks': len(chunks),
                        'status': 'split'
                    }
                else:
                    stats['skipped_pdfs'] += 1
                    stats['pdfs'][pdf_file.name] = {
                        'pages': total_pages,
                        'size_mb': round(file_size_mb, 1),
                        'chunks': 0,
                        'status': 'skip'
                    }
            else:
                print(f"\n  📄 {pdf_file.name}")
                print(f"      ⏭️  Skipping ({total_pages} pages, {file_size_mb:.1f}MB - below thresholds)")
                stats['skipped_pdfs'] += 1
                stats['pdfs'][pdf_file.name] = {
                    'pages': total_pages,
                    'size_mb': round(file_size_mb, 1),
                    'chunks': 0,
                    'status': 'skip'
                }
        except Exception as e:
            print(f"\n  ❌ Error processing {pdf_file.name}: {e}")
            stats['pdfs'][pdf_file.name] = {
                'error': str(e),
                'status': 'error'
            }

    # Save mapping
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'chunks': all_chunks,
            'direct': [
                {'pdf_path': str(input_dir / name), 'pdf_name': name}
                for name, info in stats['pdfs'].items()
                if info.get('status') == 'skip'
            ]
        }, f, indent=2)

    # Update progress manager
    if progress_manager:
        progress_manager.set_step_status('split', 'completed')
        # Set list of PDFs to process
        chunk_pdfs = [Path(c['path']).stem for c in all_chunks]
        direct_pdfs = [Path(name).stem for name, info in stats['pdfs'].items() if info.get('status') == 'skip']
        progress_manager.set_pdf_list('process_chunks', chunk_pdfs)
        progress_manager.set_pdf_list('process_direct', direct_pdfs)
        progress_manager._data['statistics']['total_pdfs'] = len(chunk_pdfs) + len(direct_pdfs)
        progress_manager.save()

    print()
    print("="*80)
    print("✅ Splitting Complete")
    print("="*80)
    print(f"Total PDFs: {stats['total_pdfs']}")
    print(f"Split PDFs: {stats['split_pdfs']}")
    print(f"Skipped PDFs: {stats['skipped_pdfs']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Mapping saved to: {mapping_file}")
    print()

if __name__ == "__main__":
    sys.exit(main())
