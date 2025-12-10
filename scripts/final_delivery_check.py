#!/usr/bin/env python3
"""
Final Delivery Validation Script
Performs comprehensive checks on the final-delivery folder to ensure quality.

Checks performed:
1. Folder structure verification (expected count)
2. File completeness (MD + YAML per folder)
3. JSON remnants detection
4. Literal \n outside LaTeX context
5. YAML syntax validation
6. Markdown structure verification
"""

import os
import sys
import re
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any


class DeliveryChecker:
    def __init__(self, delivery_dir: str, expected_count: int = None, verbose: bool = False):
        self.delivery_dir = Path(delivery_dir)
        self.expected_count = expected_count
        self.verbose = verbose
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.stats = {
            'folders_checked': 0,
            'md_files_checked': 0,
            'yaml_files_checked': 0,
            'total_pages': 0,
            'total_figures': 0,
            'total_chars': 0
        }

    def log(self, msg: str, level: str = "INFO"):
        """Print log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "OK": "✅"}.get(level, "")
        print(f"[{timestamp}] {prefix} {msg}")

    def add_issue(self, folder: str, check: str, message: str):
        """Record an issue"""
        self.issues.append({
            'folder': folder,
            'check': check,
            'message': message
        })

    def add_warning(self, folder: str, check: str, message: str):
        """Record a warning"""
        self.warnings.append({
            'folder': folder,
            'check': check,
            'message': message
        })

    def check_folder_structure(self) -> bool:
        """Check 1: Verify file count (flat structure)"""
        self.log("Check 1: File count verification")

        if not self.delivery_dir.exists():
            self.add_issue("", "file_structure", f"Delivery directory does not exist: {self.delivery_dir}")
            return False

        # Flat structure: count .md files directly (excluding reports)
        md_files = [f for f in self.delivery_dir.glob('*.md')
                    if not f.name.startswith('QUALITY_REPORT')]

        self.stats['folders_checked'] = len(md_files)  # Now counts MD files instead of folders

        if self.expected_count and len(md_files) != self.expected_count:
            self.add_issue("", "file_structure",
                f"Expected {self.expected_count} MD files, found {len(md_files)}")
            return False

        self.log(f"  Found {len(md_files)} MD files", "OK")
        return True

    def check_file_completeness(self) -> bool:
        """Check 2: Each MD file has a corresponding YAML file (flat structure)"""
        self.log("Check 2: File completeness (MD + YAML pairs)")

        # Flat structure: find all .md files
        md_files = sorted([f for f in self.delivery_dir.glob('*.md')
                          if not f.name.startswith('QUALITY_REPORT')])
        all_complete = True

        for md_file in md_files:
            # Check for corresponding YAML file
            yaml_file = self.delivery_dir / f"{md_file.stem}.yaml"

            self.stats['md_files_checked'] += 1

            if not yaml_file.exists():
                self.add_issue(md_file.stem, "file_completeness", "Missing corresponding YAML file")
                all_complete = False
            else:
                self.stats['yaml_files_checked'] += 1

        if all_complete:
            self.log(f"  All {len(md_files)} MD files have corresponding YAML files", "OK")
        return all_complete

    def check_json_remnants(self) -> bool:
        """Check 3: No JSON remnants in markdown files"""
        self.log("Check 3: JSON remnants detection")

        # Flat structure: find all .md files directly
        md_files = sorted([f for f in self.delivery_dir.glob('*.md')
                          if not f.name.startswith('QUALITY_REPORT')])
        clean = True

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
            except Exception as e:
                self.add_issue(md_file.stem, "json_remnants", f"Cannot read {md_file.name}: {e}")
                clean = False
                continue

            # Check for JSON code blocks
            json_blocks = len(re.findall(r'```json', content, re.IGNORECASE))
            if json_blocks > 0:
                self.add_issue(md_file.stem, "json_remnants",
                    f"Found {json_blocks} JSON code blocks in {md_file.name}")
                clean = False

            # Check for raw JSON structure remnants
            body_text_refs = content.count('"body_text":')
            if body_text_refs > 0:
                self.add_issue(md_file.stem, "json_remnants",
                    f"Found {body_text_refs} 'body_text' references in {md_file.name}")
                clean = False

            # Check for other JSON patterns
            chart_type_refs = content.count('"chart_type":')
            if chart_type_refs > 0:
                self.add_warning(md_file.stem, "json_remnants",
                    f"Found {chart_type_refs} 'chart_type' references in {md_file.name}")

        if clean:
            self.log("  No JSON remnants found", "OK")
        return clean

    def check_literal_newlines(self) -> bool:
        """Check 4: Literal \\n outside LaTeX context"""
        self.log("Check 4: Literal newline detection")

        # Flat structure: find all .md files directly
        md_files = sorted([f for f in self.delivery_dir.glob('*.md')
                          if not f.name.startswith('QUALITY_REPORT')])
        clean = True

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
            except:
                continue

            # Find literal \n outside of LaTeX contexts
            # LaTeX contexts: $...$, $$...$$, \[...\], \(...\)

            # Remove LaTeX blocks first for checking
            temp_content = content
            # Remove display math
            temp_content = re.sub(r'\$\$.*?\$\$', '', temp_content, flags=re.DOTALL)
            # Remove inline math
            temp_content = re.sub(r'\$[^$]+\$', '', temp_content)
            # Remove \[...\] blocks
            temp_content = re.sub(r'\\\[.*?\\\]', '', temp_content, flags=re.DOTALL)

            # Now check for literal \n in remaining content
            literal_n_count = temp_content.count('\\n')

            # Allow some in code blocks
            code_blocks = len(re.findall(r'```.*?```', temp_content, flags=re.DOTALL))

            if literal_n_count > code_blocks * 10:  # Threshold
                self.add_warning(md_file.stem, "literal_newlines",
                    f"Found {literal_n_count} literal \\n in {md_file.name} (may need review)")

        if clean:
            self.log("  No problematic literal newlines found", "OK")
        return clean

    def check_yaml_syntax(self) -> bool:
        """Check 5: YAML files are valid"""
        self.log("Check 5: YAML syntax validation")

        # Flat structure: find all .yaml files directly (excluding reports)
        yaml_files = sorted([f for f in self.delivery_dir.glob('*.yaml')
                            if not f.name.startswith('VALIDATION_REPORT')])
        valid = True

        for yaml_file in yaml_files:
            try:
                content = yaml_file.read_text(encoding='utf-8')
                data = yaml.safe_load(content)

                # Check if data is valid (not None and not empty for main files)
                if data is None and yaml_file.stat().st_size > 10:
                    self.add_warning(yaml_file.stem, "yaml_syntax",
                        f"YAML file parses to None: {yaml_file.name}")

            except yaml.YAMLError as e:
                self.add_issue(yaml_file.stem, "yaml_syntax",
                    f"Invalid YAML in {yaml_file.name}: {e}")
                valid = False
            except Exception as e:
                self.add_issue(yaml_file.stem, "yaml_syntax",
                    f"Cannot read {yaml_file.name}: {e}")
                valid = False

        if valid:
            self.log("  All YAML files are valid", "OK")
        return valid

    def check_markdown_structure(self) -> bool:
        """Check 6: Markdown structure verification"""
        self.log("Check 6: Markdown structure verification")

        # Flat structure: find all .md files directly
        md_files = sorted([f for f in self.delivery_dir.glob('*.md')
                          if not f.name.startswith('QUALITY_REPORT')])
        valid = True

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
            except:
                continue

            # Count page headers
            pages = len(re.findall(r'^## Page \d+', content, re.MULTILINE))

            # Count figures
            figures = len(re.findall(r'### Fig \d+:', content))

            # File size
            file_size = len(content)

            # Update stats
            self.stats['total_pages'] += pages
            self.stats['total_figures'] += figures
            self.stats['total_chars'] += file_size

            # Warnings for potentially problematic files
            if pages == 0:
                self.add_warning(md_file.stem, "markdown_structure",
                    f"No page headers found in {md_file.name}")

            if file_size < 1000:
                self.add_warning(md_file.stem, "markdown_structure",
                    f"Very small file ({file_size} chars): {md_file.name}")

            if self.verbose:
                self.log(f"  {md_file.stem}: {pages} pages, {figures} figures, {file_size:,} chars")

        self.log(f"  Total: {self.stats['total_pages']} pages, {self.stats['total_figures']} figures, {self.stats['total_chars']:,} chars", "OK")
        return valid

    def run_all_checks(self) -> bool:
        """Run all checks and return overall status"""
        print("=" * 70)
        print("🔍 Final Delivery Validation (Flat Structure)")
        print("=" * 70)
        print(f"Directory: {self.delivery_dir}")
        if self.expected_count:
            print(f"Expected MD files: {self.expected_count}")
        print()

        results = []
        results.append(("Folder Structure", self.check_folder_structure()))
        results.append(("File Completeness", self.check_file_completeness()))
        results.append(("JSON Remnants", self.check_json_remnants()))
        results.append(("Literal Newlines", self.check_literal_newlines()))
        results.append(("YAML Syntax", self.check_yaml_syntax()))
        results.append(("Markdown Structure", self.check_markdown_structure()))

        # Print summary
        print()
        print("=" * 70)
        print("📊 Validation Summary")
        print("=" * 70)

        all_passed = True
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        print()
        print(f"Statistics:")
        print(f"  Folders checked: {self.stats['folders_checked']}")
        print(f"  MD files checked: {self.stats['md_files_checked']}")
        print(f"  YAML files checked: {self.stats['yaml_files_checked']}")
        print(f"  Total pages: {self.stats['total_pages']:,}")
        print(f"  Total figures: {self.stats['total_figures']:,}")
        print(f"  Total characters: {self.stats['total_chars']:,}")

        # Print issues
        if self.issues:
            print()
            print("❌ Issues Found:")
            for issue in self.issues:
                folder_str = f"[{issue['folder']}] " if issue['folder'] else ""
                print(f"  {folder_str}{issue['check']}: {issue['message']}")

        # Print warnings
        if self.warnings:
            print()
            print("⚠️  Warnings:")
            for warning in self.warnings[:10]:  # Limit to 10
                folder_str = f"[{warning['folder']}] " if warning['folder'] else ""
                print(f"  {folder_str}{warning['check']}: {warning['message']}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")

        print()
        if all_passed and not self.issues:
            print("✅ All checks passed!")
        else:
            print(f"❌ Validation failed with {len(self.issues)} issues")

        return all_passed and not self.issues

    def save_report(self, output_path: str):
        """Save validation report as JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'delivery_dir': str(self.delivery_dir),
            'expected_count': self.expected_count,
            'stats': self.stats,
            'issues': self.issues,
            'warnings': self.warnings,
            'passed': len(self.issues) == 0
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Final Delivery Validation Script")
    parser.add_argument("--delivery-dir", required=True, help="Path to final-delivery directory")
    parser.add_argument("--expected-count", type=int, help="Expected number of folders")
    parser.add_argument("--output-report", help="Save validation report to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    checker = DeliveryChecker(
        delivery_dir=args.delivery_dir,
        expected_count=args.expected_count,
        verbose=args.verbose
    )

    passed = checker.run_all_checks()

    if args.output_report:
        checker.save_report(args.output_report)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
