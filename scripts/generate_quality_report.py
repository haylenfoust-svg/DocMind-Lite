#!/usr/bin/env python3
"""
Quality Report Generator - DocMind Lite
Generates QUALITY_REPORT.md with comprehensive quality assessment

Features:
- Executive summary (success rate/failure rate/duration/cost estimate)
- Quality score (0-100)
- Content analysis (empty pages/short content/long content distribution)
- Element statistics (tables/formulas/images count)
- Issue list (failure reasons/manual review needed)
- MARCO compliance check
"""

import argparse
import json
import os
import re
import statistics
import sys
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PageStats:
    """Single page statistics"""
    page_num: int
    char_count: int
    has_table: bool = False
    has_formula: bool = False
    has_image: bool = False
    has_code: bool = False
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class PDFStats:
    """Single PDF statistics"""
    name: str
    md_path: str
    total_pages: int = 0
    char_count: int = 0
    page_stats: List[PageStats] = field(default_factory=list)

    # Content distribution
    empty_pages: int = 0      # <10 chars
    short_pages: int = 0      # 10-99 chars
    medium_pages: int = 0     # 100-499 chars
    long_pages: int = 0       # >=500 chars

    # Element statistics
    table_count: int = 0
    formula_count: int = 0
    image_count: int = 0
    code_block_count: int = 0

    # Quality score
    quality_score: float = 0.0

    # Issues
    issues: List[str] = field(default_factory=list)
    needs_review: bool = False

    # MARCO compliance
    has_yaml: bool = False
    marco_compliant: bool = False


@dataclass
class OverallStats:
    """Overall statistics"""
    total_pdfs: int = 0
    successful_pdfs: int = 0
    failed_pdfs: int = 0

    total_pages: int = 0
    successful_pages: int = 0

    total_chars: int = 0

    # Content distribution
    empty_pages: int = 0
    short_pages: int = 0
    medium_pages: int = 0
    long_pages: int = 0

    # Element statistics
    total_tables: int = 0
    total_formulas: int = 0
    total_images: int = 0
    total_code_blocks: int = 0

    # Time and cost
    processing_time_seconds: int = 0
    estimated_cost_rmb: float = 0.0

    # Resource usage
    resource_usage: Dict = field(default_factory=dict)

    # Quality score
    overall_quality_score: float = 0.0
    marco_compliance_rate: float = 0.0

    # PDF details
    pdf_stats: List[PDFStats] = field(default_factory=list)
    failed_list: List[Dict] = field(default_factory=list)
    review_needed: List[str] = field(default_factory=list)

    # Retry statistics (from retry_failures.yaml)
    retry_attempted: int = 0
    retry_successful: int = 0
    permanent_failures: List[Dict] = field(default_factory=list)
    skipped_pages: List[Dict] = field(default_factory=list)


class QualityReportGenerator:
    """Quality Report Generator"""

    def __init__(self, input_dir: str, output_path: str = None, progress_file: str = None):
        """
        Initialize

        Args:
            input_dir: final-delivery directory
            output_path: Report output path
            progress_file: progress.json path (optional, for time info)
        """
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path) if output_path else self.input_dir / "QUALITY_REPORT.md"
        self.progress_file = Path(progress_file) if progress_file else None
        self.stats = OverallStats()

    def analyze_page_content(self, content: str, page_num: int) -> PageStats:
        """Analyze single page content"""
        stats = PageStats(page_num=page_num, char_count=len(content))

        # Detect tables (Markdown table syntax)
        if re.search(r'\|.*\|.*\|', content) and re.search(r'\|-+\|', content):
            stats.has_table = True

        # Detect formulas (LaTeX)
        if re.search(r'\$\$.+?\$\$', content, re.DOTALL) or re.search(r'\$[^$]+\$', content):
            stats.has_formula = True

        # Detect image references
        if re.search(r'!\[.*?\]\(.*?\)', content) or re.search(r'<figure>', content, re.IGNORECASE):
            stats.has_image = True

        # Detect code blocks
        if re.search(r'```[\s\S]*?```', content):
            stats.has_code = True

        # Quality issue detection
        if len(content.strip()) < 10:
            stats.quality_issues.append("Blank page or very little content")
        elif len(content.strip()) < 50:
            stats.quality_issues.append("Content too short, possible recognition failure")

        # Detect possible garbled text
        garbled_ratio = len(re.findall(r'[�\ufffd]', content)) / max(len(content), 1)
        if garbled_ratio > 0.05:
            stats.quality_issues.append(f"Possible garbled text ({garbled_ratio:.1%})")

        # Detect consecutive repeated characters (possible recognition error)
        if re.search(r'(.)\1{10,}', content):
            stats.quality_issues.append("Abnormal repeated characters detected")

        return stats

    def analyze_pdf(self, md_file: Path) -> Optional[PDFStats]:
        """Analyze single PDF output (flat structure: directly pass .md file path)"""
        pdf_name = md_file.stem  # Get PDF name from filename

        stats = PDFStats(name=pdf_name, md_path=str(md_file))

        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            stats.issues.append(f"Unable to read file: {e}")
            return stats

        stats.char_count = len(content)

        # Split by pages
        pages = re.split(r'^## Page \d+', content, flags=re.MULTILINE)
        pages = [p.strip() for p in pages if p.strip()]

        # Skip title section
        if pages and pages[0].startswith('#'):
            pages = pages[1:]

        stats.total_pages = len(pages)

        # Analyze each page
        for i, page_content in enumerate(pages, 1):
            page_stats = self.analyze_page_content(page_content, i)
            stats.page_stats.append(page_stats)

            # Cumulative statistics
            char_len = page_stats.char_count
            if char_len < 10:
                stats.empty_pages += 1
            elif char_len < 100:
                stats.short_pages += 1
            elif char_len < 500:
                stats.medium_pages += 1
            else:
                stats.long_pages += 1

            if page_stats.has_table:
                stats.table_count += 1
            if page_stats.has_formula:
                stats.formula_count += 1
            if page_stats.has_image:
                stats.image_count += 1
            if page_stats.has_code:
                stats.code_block_count += 1

            if page_stats.quality_issues:
                stats.issues.extend([f"Page {i}: {issue}" for issue in page_stats.quality_issues])

        # Check YAML file (flat structure: same-named .yaml file in same directory)
        yaml_file = md_file.parent / f"{md_file.stem}.yaml"
        stats.has_yaml = yaml_file.exists()

        # MARCO compliance check
        stats.marco_compliant = self._check_marco_compliance(content, stats.has_yaml)

        # Calculate quality score
        if stats.total_pages > 0:
            # Valid content ratio (>=100 chars)
            valid_ratio = (stats.medium_pages + stats.long_pages) / stats.total_pages
            stats.quality_score = valid_ratio * 100

            # If too many blank pages or quality issues, mark for review
            if stats.empty_pages / stats.total_pages > 0.2 or len(stats.issues) > stats.total_pages * 0.1:
                stats.needs_review = True

        return stats

    def _check_marco_compliance(self, content: str, has_yaml: bool) -> bool:
        """Check MARCO compliance"""
        checks = [
            has_yaml,  # Has YAML file
            bool(re.search(r'^# ', content, re.MULTILINE)),  # Has title
            bool(re.search(r'^## Page \d+', content, re.MULTILINE)),  # Has page markers
        ]
        return all(checks)

    def load_progress_info(self) -> Dict:
        """Load processing info from progress.json"""
        if not self.progress_file or not self.progress_file.exists():
            return {}

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def calculate_estimated_cost(self, total_pages: int) -> float:
        """Estimate processing cost (based on qwen-vl-plus pricing)"""
        # qwen-vl-plus: ~0.008 RMB/1k tokens
        # Estimate ~2000 tokens per page (input image + output text)
        tokens_per_page = 2000
        price_per_1k_tokens = 0.008

        total_tokens = total_pages * tokens_per_page
        cost = (total_tokens / 1000) * price_per_1k_tokens
        return round(cost, 2)

    def generate(self) -> OverallStats:
        """Generate quality report"""
        print(f"📊 Analyzing directory: {self.input_dir}")

        # Flat structure: directly find all .md files (exclude QUALITY_REPORT.md)
        md_files = [f for f in self.input_dir.glob("*.md")
                    if not f.name.startswith('QUALITY_REPORT')]

        self.stats.total_pdfs = len(md_files)

        for md_file in sorted(md_files):
            pdf_stats = self.analyze_pdf(md_file)
            if pdf_stats:
                if pdf_stats.total_pages > 0:
                    self.stats.successful_pdfs += 1
                    self.stats.pdf_stats.append(pdf_stats)

                    # Cumulative statistics
                    self.stats.total_pages += pdf_stats.total_pages
                    self.stats.total_chars += pdf_stats.char_count
                    self.stats.empty_pages += pdf_stats.empty_pages
                    self.stats.short_pages += pdf_stats.short_pages
                    self.stats.medium_pages += pdf_stats.medium_pages
                    self.stats.long_pages += pdf_stats.long_pages
                    self.stats.total_tables += pdf_stats.table_count
                    self.stats.total_formulas += pdf_stats.formula_count
                    self.stats.total_images += pdf_stats.image_count
                    self.stats.total_code_blocks += pdf_stats.code_block_count

                    if pdf_stats.needs_review:
                        self.stats.review_needed.append(pdf_stats.name)
                else:
                    self.stats.failed_pdfs += 1
                    self.stats.failed_list.append({
                        'name': pdf_stats.name,
                        'reason': 'Output empty or unable to parse'
                    })
            else:
                self.stats.failed_pdfs += 1
                self.stats.failed_list.append({
                    'name': md_file.stem,
                    'reason': 'Unable to parse .md file'
                })

        # Calculate overall quality score
        if self.stats.total_pages > 0:
            valid_ratio = (self.stats.medium_pages + self.stats.long_pages) / self.stats.total_pages
            self.stats.overall_quality_score = round(valid_ratio * 100, 1)

        # MARCO compliance rate
        marco_count = sum(1 for p in self.stats.pdf_stats if p.marco_compliant)
        if self.stats.successful_pdfs > 0:
            self.stats.marco_compliance_rate = round(marco_count / self.stats.successful_pdfs * 100, 1)

        # Load processing time
        progress_info = self.load_progress_info()
        if progress_info:
            started = progress_info.get('started_at', '')
            completed = progress_info.get('completed_at', progress_info.get('updated_at', ''))
            if started and completed:
                try:
                    start_time = datetime.fromisoformat(started)
                    end_time = datetime.fromisoformat(completed)
                    self.stats.processing_time_seconds = int((end_time - start_time).total_seconds())
                except:
                    pass

        # Estimate cost
        self.stats.estimated_cost_rmb = self.calculate_estimated_cost(self.stats.total_pages)

        # Load resource usage data
        if progress_info and 'resource_usage' in progress_info:
            self.stats.resource_usage = progress_info['resource_usage']

        # Load retry failure statistics
        self.load_retry_failures()

        return self.stats

    def load_retry_failures(self):
        """Load retry failure statistics (from retry_failures.yaml)"""
        # Try to find retry_failures.yaml in chunks directory
        chunks_dir = self.input_dir.parent / "output" / "chunks"
        retry_file = chunks_dir / "retry_failures.yaml"

        if not retry_file.exists():
            # Also check in input_dir parent
            retry_file = self.input_dir.parent / "output" / "chunks" / "retry_failures.yaml"

        if not retry_file.exists():
            return

        try:
            with open(retry_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data:
                return

            self.stats.retry_attempted = data.get('attempted_retries', 0)
            self.stats.retry_successful = data.get('successful_retries', 0)
            self.stats.permanent_failures = data.get('permanent_failures', [])
            self.stats.skipped_pages = data.get('skipped_pages', [])

        except Exception as e:
            print(f"  Warning: Could not load retry failures: {e}")

    def format_time(self, seconds: int) -> str:
        """Format time duration"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def get_quality_rating(self, score: float) -> Tuple[str, str]:
        """Get quality rating"""
        if score >= 90:
            return "Excellent", "🌟"
        elif score >= 80:
            return "Good", "✅"
        elif score >= 60:
            return "Fair", "⚠️"
        elif score >= 40:
            return "Poor", "❌"
        else:
            return "Very Poor", "🚫"

    def format_resource_section(self) -> str:
        """Format resource usage section"""
        ru = self.stats.resource_usage
        if not ru:
            return """
## 8. Resource Usage Statistics

> Resource monitoring: Not enabled or no data

"""

        cpu = ru.get('cpu', {})
        mem = ru.get('memory', {})
        disk = ru.get('disk_io', {})
        net = ru.get('network_io', {})
        sample_count = ru.get('sample_count', 0)
        interval = ru.get('sample_interval_sec', 0)

        return f"""
## 8. Resource Usage Statistics

> Samples: {sample_count} | Interval: {interval}s | Total monitoring: {self.format_time(sample_count * interval)}

| Metric | Average | Peak |
|--------|---------|------|
| CPU Usage | {cpu.get('average_percent', '-')}% | {cpu.get('peak_percent', '-')}% |
| Process Memory | {mem.get('average_process_mb', '-')} MB | {mem.get('peak_process_mb', '-')} MB |
| System Memory | - | {mem.get('peak_system_percent', '-')}% (total {mem.get('total_system_gb', '-')} GB) |
| Disk Read | {disk.get('average_read_mb_s', '-')} MB/s | - |
| Disk Write | {disk.get('average_write_mb_s', '-')} MB/s | - |
| Network Upload | {net.get('average_sent_kb_s', '-')} KB/s | - |
| Network Download | {net.get('average_recv_kb_s', '-')} KB/s | - |

"""

    def render_report(self) -> str:
        """Render Markdown report"""
        s = self.stats
        rating, emoji = self.get_quality_rating(s.overall_quality_score)

        report = f"""# DocMind Quality Report

> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
>
> Input directory: `{self.input_dir}`

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total PDFs | {s.total_pdfs} |
| Successfully processed | {s.successful_pdfs} ({s.successful_pdfs * 100 / max(s.total_pdfs, 1):.1f}%) |
| Failed | {s.failed_pdfs} |
| Total pages | {s.total_pages:,} |
| Total characters | {s.total_chars:,} |
| Processing time | {self.format_time(s.processing_time_seconds) if s.processing_time_seconds else 'Unknown'} |
| Estimated cost | ¥{s.estimated_cost_rmb:.2f} |

---

## 2. Quality Score

### Overall Score: {s.overall_quality_score:.1f}/100 {emoji} ({rating})

| Metric | Score | Description |
|--------|-------|-------------|
| Valid content rate | {s.overall_quality_score:.1f}% | Pages with 100+ chars |
| MARCO compliance | {s.marco_compliance_rate:.1f}% | MARCO format compliant |

#### Rating Scale:
- 90-100: Excellent 🌟
- 80-89: Good ✅
- 60-79: Fair ⚠️
- 40-59: Poor ❌
- 0-39: Very Poor 🚫

---

## 3. Content Distribution

| Category | Pages | Ratio | Description |
|----------|-------|-------|-------------|
| Empty/Minimal | {s.empty_pages} | {s.empty_pages * 100 / max(s.total_pages, 1):.1f}% | <10 chars |
| Short content | {s.short_pages} | {s.short_pages * 100 / max(s.total_pages, 1):.1f}% | 10-99 chars |
| Medium content | {s.medium_pages} | {s.medium_pages * 100 / max(s.total_pages, 1):.1f}% | 100-499 chars |
| Long content | {s.long_pages} | {s.long_pages * 100 / max(s.total_pages, 1):.1f}% | ≥500 chars |

```
Content Distribution Visualization:
Empty/Minimal [{"█" * int(s.empty_pages * 20 / max(s.total_pages, 1))}{"░" * (20 - int(s.empty_pages * 20 / max(s.total_pages, 1)))}] {s.empty_pages * 100 / max(s.total_pages, 1):.1f}%
Short         [{"█" * int(s.short_pages * 20 / max(s.total_pages, 1))}{"░" * (20 - int(s.short_pages * 20 / max(s.total_pages, 1)))}] {s.short_pages * 100 / max(s.total_pages, 1):.1f}%
Medium        [{"█" * int(s.medium_pages * 20 / max(s.total_pages, 1))}{"░" * (20 - int(s.medium_pages * 20 / max(s.total_pages, 1)))}] {s.medium_pages * 100 / max(s.total_pages, 1):.1f}%
Long          [{"█" * int(s.long_pages * 20 / max(s.total_pages, 1))}{"░" * (20 - int(s.long_pages * 20 / max(s.total_pages, 1)))}] {s.long_pages * 100 / max(s.total_pages, 1):.1f}%
```

---

## 4. Element Statistics

| Type | Count |
|------|-------|
| Tables | {s.total_tables} |
| Formulas (LaTeX) | {s.total_formulas} |
| Image references | {s.total_images} |
| Code blocks | {s.total_code_blocks} |

---

## 5. Issue List

### 5.1 Failed PDFs ({len(s.failed_list)})
"""

        if s.failed_list:
            report += "\n| PDF Name | Failure Reason |\n|----------|----------------|\n"
            for item in s.failed_list:
                report += f"| {item['name'][:50]}... | {item['reason']} |\n"
        else:
            report += "\n✅ No failed PDFs\n"

        report += f"""
### 5.2 Needs Manual Review ({len(s.review_needed)})
"""

        if s.review_needed:
            report += "\nThe following PDFs have many blank pages or quality issues, manual review recommended:\n\n"
            for name in s.review_needed[:20]:  # Show at most 20
                report += f"- {name}\n"
            if len(s.review_needed) > 20:
                report += f"\n... {len(s.review_needed)} total need review\n"
        else:
            report += "\n✅ No PDFs need manual review\n"

        # 5.3 Retry statistics and permanent failures
        total_permanent = len(s.permanent_failures) + len(s.skipped_pages)
        report += f"""
### 5.3 Page Retry Statistics

| Metric | Count |
|--------|-------|
| Retry attempts | {s.retry_attempted} |
| Retry successful | {s.retry_successful} |
| Permanent failures | {len(s.permanent_failures)} |
| Skipped (content filter) | {len(s.skipped_pages)} |

"""

        if s.permanent_failures:
            report += "#### Permanent Failure Pages (need manual processing)\n\n"
            report += "| File | Page | Error Reason |\n|------|------|-------------|\n"
            for failure in s.permanent_failures[:30]:  # Show at most 30
                chunk = failure.get('chunk', 'Unknown')[:40]
                page = failure.get('page', '?')
                error = failure.get('error', 'Unknown')[:50]
                report += f"| {chunk}... | {page} | {error}... |\n"
            if len(s.permanent_failures) > 30:
                report += f"\n... {len(s.permanent_failures)} total permanent failures\n"

        if s.skipped_pages:
            report += "\n#### Skipped Pages (content filter blocked)\n\n"
            report += "The following pages were blocked by content filter and cannot be recovered by retry:\n\n"
            report += "| File | Page |\n|------|------|\n"
            for skip in s.skipped_pages[:20]:  # Show at most 20
                chunk = skip.get('chunk', 'Unknown')[:40]
                page = skip.get('page', '?')
                report += f"| {chunk}... | {page} |\n"
            if len(s.skipped_pages) > 20:
                report += f"\n... {len(s.skipped_pages)} total skipped\n"

        if not s.permanent_failures and not s.skipped_pages and s.retry_attempted == 0:
            report += "\n✅ No failed pages need retry\n"

        report += """
---

## 6. Detailed Report (sorted by quality score)

| PDF Name | Pages | Quality | Empty | Tables | Formulas | MARCO |
|----------|-------|---------|-------|--------|----------|-------|
"""

        # Sort by quality score
        sorted_pdfs = sorted(s.pdf_stats, key=lambda x: x.quality_score, reverse=True)
        for pdf in sorted_pdfs:
            name_display = pdf.name[:40] + "..." if len(pdf.name) > 40 else pdf.name
            marco_mark = "✅" if pdf.marco_compliant else "❌"
            report += f"| {name_display} | {pdf.total_pages} | {pdf.quality_score:.0f} | {pdf.empty_pages} | {pdf.table_count} | {pdf.formula_count} | {marco_mark} |\n"

        report += f"""
---

## 7. Recommendations

"""

        # Give recommendations based on analysis results
        suggestions = []

        if s.overall_quality_score < 60:
            suggestions.append("⚠️ Overall quality score is low, recommend checking OCR/LLM processing")

        if s.empty_pages / max(s.total_pages, 1) > 0.1:
            suggestions.append(f"⚠️ High ratio of blank pages ({s.empty_pages * 100 / max(s.total_pages, 1):.1f}%), possible scan quality issues or recognition failures")

        if s.failed_pdfs > 0:
            suggestions.append(f"❌ {s.failed_pdfs} PDFs failed processing, recommend checking logs for root cause")

        if s.marco_compliance_rate < 100:
            suggestions.append(f"📝 MARCO compliance rate {s.marco_compliance_rate:.0f}%, some files missing required metadata")

        if len(s.review_needed) > 0:
            suggestions.append(f"👀 {len(s.review_needed)} PDFs need manual review")

        if not suggestions:
            suggestions.append("✅ Processing quality is good, no special recommendations")

        for suggestion in suggestions:
            report += f"- {suggestion}\n"

        report += """
---
"""

        # 添加资源使用统计（Section 8）
        report += self.format_resource_section()

        report += """---

*Report generated by DocMind Lite Quality Report Generator*
"""

        return report

    def save_report(self, report: str):
        """Save report"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Report saved: {self.output_path}")

    def run(self) -> str:
        """Run complete workflow"""
        self.generate()
        report = self.render_report()
        self.save_report(report)
        return report


def main():
    parser = argparse.ArgumentParser(
        description='DocMind Quality Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./final-delivery
  %(prog)s --input ./final-delivery --output ./QUALITY_REPORT.md
  %(prog)s --input ./final-delivery --progress ./progress.json
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='final-delivery directory path'
    )

    parser.add_argument(
        '--output', '-o',
        help='Report output path (default: <input>/QUALITY_REPORT.md)'
    )

    parser.add_argument(
        '--progress', '-p',
        help='progress.json file path (optional, for processing time)'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Also output JSON format report'
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Generate report
    generator = QualityReportGenerator(
        input_dir=str(input_dir),
        output_path=args.output,
        progress_file=args.progress
    )

    report = generator.run()

    # Output summary to terminal
    s = generator.stats
    rating, emoji = generator.get_quality_rating(s.overall_quality_score)

    print("\n" + "=" * 60)
    print("📊 Quality Report Summary")
    print("=" * 60)
    print(f"  Total PDFs: {s.total_pdfs} | Success: {s.successful_pdfs} | Failed: {s.failed_pdfs}")
    print(f"  Total pages: {s.total_pages:,} | Total chars: {s.total_chars:,}")
    print(f"  Quality score: {s.overall_quality_score:.1f}/100 {emoji} ({rating})")
    print(f"  MARCO compliance: {s.marco_compliance_rate:.1f}%")
    if s.review_needed:
        print(f"  Needs review: {len(s.review_needed)} PDFs")
    print("=" * 60)

    # Optional: JSON output
    if args.json:
        json_path = generator.output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_pdfs': s.total_pdfs,
                'successful_pdfs': s.successful_pdfs,
                'failed_pdfs': s.failed_pdfs,
                'total_pages': s.total_pages,
                'overall_quality_score': s.overall_quality_score,
                'marco_compliance_rate': s.marco_compliance_rate,
                'content_distribution': {
                    'empty': s.empty_pages,
                    'short': s.short_pages,
                    'medium': s.medium_pages,
                    'long': s.long_pages
                },
                'elements': {
                    'tables': s.total_tables,
                    'formulas': s.total_formulas,
                    'images': s.total_images,
                    'code_blocks': s.total_code_blocks
                },
                'resource_usage': s.resource_usage,
                'failed_list': s.failed_list,
                'review_needed': s.review_needed
            }, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON report: {json_path}")


if __name__ == "__main__":
    main()
