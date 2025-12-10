#!/usr/bin/env python3
"""
Task Report Generator

Usage:
    python3 generate_report.py --task-name "my_batch_001"

Features:
    - Collects all reports from task output
    - Generates quality report, cost breakdown, validation results
    - Saves all reports to reports/{task_name}/ directory
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class TaskReportGenerator:
    """Task Report Generator"""

    # Qwen-VL-Max pricing (2024 prices)
    # Note: This is official pricing, actual costs may differ with discounts
    PRICE_INPUT_PER_1K = 0.02   # ¥0.02/1000 input tokens
    PRICE_OUTPUT_PER_1K = 0.02  # ¥0.02/1000 output tokens

    def __init__(self, base_dir: str, task_name: str):
        self.base_dir = Path(base_dir)
        self.task_name = task_name
        self.reports_dir = self.base_dir / "reports" / task_name
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def collect_batch_reports(self) -> List[Dict]:
        """Collect all batch_report JSON files"""
        reports = []

        # Search output/chunks directory
        chunks_dir = self.base_dir / "output" / "chunks"
        if chunks_dir.exists():
            for f in chunks_dir.glob("batch_report_*.json"):
                try:
                    with open(f, encoding='utf-8') as fp:
                        reports.append(json.load(fp))
                except Exception as e:
                    print(f"  Warning: Cannot read {f}: {e}")

        # Search other possible locations
        for pattern in ["**/batch_report_*.json", "**/chunks/batch_report_*.json"]:
            for f in self.base_dir.glob(pattern):
                if "output/chunks" not in str(f):  # Avoid duplicates
                    try:
                        with open(f, encoding='utf-8') as fp:
                            data = json.load(fp)
                            if data not in reports:
                                reports.append(data)
                    except Exception:
                        pass

        return reports

    def collect_progress_data(self) -> Optional[Dict]:
        """Collect progress.json data"""
        progress_file = self.base_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def collect_monitor_data(self) -> List[Dict]:
        """Collect monitoring data"""
        monitor_dir = self.base_dir / "monitor"
        data = []

        if monitor_dir.exists():
            # Find files matching task name
            for f in monitor_dir.glob(f"{self.task_name}*_summary.json"):
                try:
                    with open(f, encoding='utf-8') as fp:
                        data.append(json.load(fp))
                except Exception:
                    pass

            # If not found, get the latest
            if not data:
                summaries = sorted(monitor_dir.glob("*_summary.json"), reverse=True)
                if summaries:
                    try:
                        with open(summaries[0], encoding='utf-8') as fp:
                            data.append(json.load(fp))
                    except Exception:
                        pass

        return data

    def calculate_costs(self, batch_reports: List[Dict]) -> Dict:
        """Calculate cost breakdown"""
        total_input_tokens = 0
        total_output_tokens = 0
        total_pages = 0
        total_figures = 0
        total_time = 0

        pdf_details = []

        for report in batch_reports:
            if "pdfs" in report:
                for pdf in report["pdfs"]:
                    if pdf.get("success"):
                        proc = pdf.get("processing", {})
                        info = pdf.get("pdf_info", {})

                        tokens = proc.get("tokens", 0)
                        pages = info.get("pages", 0)
                        figures = info.get("figures", 0)
                        time_sec = proc.get("time", 0)

                        # Estimate input tokens (mainly images)
                        # Each page image is about 1500 tokens (150 DPI, A4)
                        est_input_tokens = pages * 1500
                        # Recorded tokens are treated as output tokens
                        output_tokens = tokens

                        total_input_tokens += est_input_tokens
                        total_output_tokens += output_tokens
                        total_pages += pages
                        total_figures += figures
                        total_time += time_sec

                        pdf_details.append({
                            "name": info.get("name", "unknown"),
                            "pages": pages,
                            "figures": figures,
                            "input_tokens_est": est_input_tokens,
                            "output_tokens": output_tokens,
                            "time_sec": round(time_sec, 1),
                        })

            # Also check summary
            if "summary" in report:
                summary = report["summary"]
                if total_pages == 0:
                    total_pages = summary.get("total_pages", 0)
                    total_figures = summary.get("total_figures", 0)
                    total_time = summary.get("total_time", 0)

        # Calculate costs
        input_cost = total_input_tokens * self.PRICE_INPUT_PER_1K / 1000
        output_cost = total_output_tokens * self.PRICE_OUTPUT_PER_1K / 1000
        total_cost = input_cost + output_cost

        # Cost recorded in code (may be lower)
        recorded_cost = sum(
            r.get("summary", {}).get("total_cost_cny", 0)
            for r in batch_reports
        )

        return {
            "task_name": self.task_name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_pages": total_pages,
                "total_figures": total_figures,
                "total_time_sec": round(total_time, 1),
                "total_time_formatted": self._format_duration(total_time),
            },
            "tokens": {
                "input_tokens_estimated": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            "cost_analysis": {
                "pricing": {
                    "input_per_1k": self.PRICE_INPUT_PER_1K,
                    "output_per_1k": self.PRICE_OUTPUT_PER_1K,
                    "model": "qwen-vl-max-latest",
                },
                "calculated": {
                    "input_cost_cny": round(input_cost, 2),
                    "output_cost_cny": round(output_cost, 2),
                    "total_cost_cny": round(total_cost, 2),
                },
                "recorded_in_code": round(recorded_cost, 2),
                "note": "calculated is based on official pricing estimate, recorded_in_code is the value recorded in code, actual bills may differ due to discounts",
            },
            "per_page_metrics": {
                "cost_per_page": round(total_cost / max(total_pages, 1), 4),
                "tokens_per_page": round((total_input_tokens + total_output_tokens) / max(total_pages, 1), 0),
                "time_per_page_sec": round(total_time / max(total_pages, 1), 2),
            },
            "pdf_details": pdf_details[:50],  # Keep only first 50 to avoid large files
        }

    def generate_quality_report(self, batch_reports: List[Dict], progress: Optional[Dict]) -> str:
        """Generate quality report Markdown"""

        # Collect statistics
        total_pages = 0
        total_figures = 0
        total_chars = 0
        failed_pages = 0
        validation_stats = {
            "total_tables": 0,
            "total_formulas": 0,
        }

        for report in batch_reports:
            if "summary" in report:
                s = report["summary"]
                total_pages += s.get("total_pages", 0)
                total_figures += s.get("total_figures", 0)

            if "pdfs" in report:
                for pdf in report["pdfs"]:
                    val = pdf.get("validation", {})
                    validation_stats["total_tables"] += val.get("total_tables_detected", 0)
                    validation_stats["total_formulas"] += val.get("total_formulas_detected", 0)

        # Get failure info from progress
        if progress:
            pdfs_data = progress.get("pdfs", {})
            for pdf_name, pdf_info in pdfs_data.items():
                failed_pages += len(pdf_info.get("failed_pages", []))

        # Generate report
        report = f"""# Quality Report: {self.task_name}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Summary

| Metric | Value |
|--------|-------|
| Total Pages | {total_pages:,} |
| Total Figures | {total_figures:,} |
| Tables | {validation_stats['total_tables']:,} |
| Formulas | {validation_stats['total_formulas']:,} |
| Failed Pages | {failed_pages} |
| Success Rate | {((total_pages - failed_pages) / max(total_pages, 1) * 100):.1f}% |

## Validation Results

"""

        # Add PDF validation details
        if batch_reports:
            report += "### PDF Processing Status\n\n"
            report += "| PDF | Pages | Figures | Status |\n"
            report += "|-----|-------|---------|--------|\n"

            count = 0
            for br in batch_reports:
                for pdf in br.get("pdfs", []):
                    if count >= 30:  # Limit rows
                        report += "| ... | ... | ... | ... |\n"
                        break
                    info = pdf.get("pdf_info", {})
                    status = "✅" if pdf.get("success") else "❌"
                    report += f"| {info.get('name', 'unknown')[:40]} | {info.get('pages', 0)} | {info.get('figures', 0)} | {status} |\n"
                    count += 1

        report += """
## Notes

- This report was auto-generated by `generate_report.py`
- Detailed cost data in `cost_summary.json`
- Validation data in `validation_result.json`
"""

        return report

    def generate_validation_result(self, batch_reports: List[Dict]) -> Dict:
        """Generate validation result JSON"""
        results = {
            "task_name": self.task_name,
            "generated_at": datetime.now().isoformat(),
            "pdfs": [],
        }

        for report in batch_reports:
            for pdf in report.get("pdfs", []):
                results["pdfs"].append({
                    "name": pdf.get("pdf_info", {}).get("name", "unknown"),
                    "success": pdf.get("success", False),
                    "validation": pdf.get("validation", {}),
                })

        # Summary statistics
        total = len(results["pdfs"])
        success = sum(1 for p in results["pdfs"] if p["success"])

        results["summary"] = {
            "total_pdfs": total,
            "successful": success,
            "failed": total - success,
            "success_rate": round(success / max(total, 1) * 100, 1),
        }

        return results

    def copy_processing_log(self):
        """Copy processing logs to report directory"""
        log_sources = [
            self.base_dir / "logs" / "chunks_processing.log",
            self.base_dir / "logs" / "direct_processing.log",
        ]

        combined_log = self.reports_dir / "processing.log"

        with open(combined_log, 'w', encoding='utf-8') as out:
            out.write(f"# Processing Log for {self.task_name}\n")
            out.write(f"# Generated: {datetime.now().isoformat()}\n\n")

            for log_file in log_sources:
                if log_file.exists():
                    out.write(f"\n{'='*60}\n")
                    out.write(f"# Source: {log_file.name}\n")
                    out.write(f"{'='*60}\n\n")
                    try:
                        # Only read last 1000 lines
                        with open(log_file, encoding='utf-8') as f:
                            lines = f.readlines()
                            out.writelines(lines[-1000:])
                    except Exception as e:
                        out.write(f"Error reading log: {e}\n")

    def generate_all(self) -> Dict[str, str]:
        """Generate all reports, return file paths"""
        print(f"\n📊 Generating task report: {self.task_name}")
        print(f"   Output directory: {self.reports_dir}")

        # Collect data
        print("   Collecting batch reports...")
        batch_reports = self.collect_batch_reports()
        print(f"   Found {len(batch_reports)} batch report(s)")

        print("   Collecting progress data...")
        progress = self.collect_progress_data()

        print("   Collecting monitoring data...")
        monitor_data = self.collect_monitor_data()

        generated_files = {}

        # 1. Cost breakdown
        print("   Generating cost breakdown...")
        cost_data = self.calculate_costs(batch_reports)
        cost_file = self.reports_dir / "cost_summary.json"
        with open(cost_file, 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        generated_files["cost_summary"] = str(cost_file)
        print(f"      ✅ {cost_file.name}")

        # 2. Quality report
        print("   Generating quality report...")
        quality_md = self.generate_quality_report(batch_reports, progress)
        quality_file = self.reports_dir / "quality_report.md"
        with open(quality_file, 'w', encoding='utf-8') as f:
            f.write(quality_md)
        generated_files["quality_report"] = str(quality_file)
        print(f"      ✅ {quality_file.name}")

        # 3. Validation result
        print("   Generating validation result...")
        validation = self.generate_validation_result(batch_reports)
        validation_file = self.reports_dir / "validation_result.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation, f, indent=2, ensure_ascii=False)
        generated_files["validation_result"] = str(validation_file)
        print(f"      ✅ {validation_file.name}")

        # 4. Processing log
        print("   Copying processing log...")
        self.copy_processing_log()
        generated_files["processing_log"] = str(self.reports_dir / "processing.log")
        print(f"      ✅ processing.log")

        # Print summary
        print(f"\n{'='*60}")
        print(f"📈 Cost Summary")
        print(f"{'='*60}")
        print(f"   Total pages: {cost_data['summary']['total_pages']:,}")
        print(f"   Processing time: {cost_data['summary']['total_time_formatted']}")
        print(f"   Estimated cost: ¥{cost_data['cost_analysis']['calculated']['total_cost_cny']:.2f}")
        print(f"   Recorded in code: ¥{cost_data['cost_analysis']['recorded_in_code']:.2f}")
        print(f"   Cost per page: ¥{cost_data['per_page_metrics']['cost_per_page']:.4f}")
        print(f"{'='*60}\n")

        return generated_files

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(description="Generate task report")
    parser.add_argument("--base-dir", "-d",
                       default=str(Path(__file__).parent.parent),
                       help="DocMind root directory")
    parser.add_argument("--task-name", "-t", required=True,
                       help="Task name (e.g., my_batch_001)")
    parser.add_argument("--from-batch", "-b",
                       help="Specify batch_report.json file path")

    args = parser.parse_args()

    generator = TaskReportGenerator(args.base_dir, args.task_name)
    files = generator.generate_all()

    print("✅ Report generation complete!")
    print("   File list:")
    for name, path in files.items():
        print(f"   - {name}: {path}")


if __name__ == "__main__":
    main()
