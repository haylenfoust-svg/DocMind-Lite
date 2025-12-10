#!/usr/bin/env python3
"""
Markdown Post-processor - DocMind Lite
Clean and optimize generated Markdown files

Features:
- Table alignment fix
- Consecutive empty lines merge
- Heading level standardization
- LaTeX formula validation
- Auto-generate Table of Contents (TOC)
- Link validity check
- Orphan heading detection
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PostProcessResult:
    """Post-processing result"""
    file_path: str
    original_size: int
    processed_size: int
    fixes_applied: List[str] = field(default_factory=list)
    toc_generated: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MarkdownPostProcessor:
    """Markdown Post-processor"""

    def __init__(self, fix_tables: bool = True, fix_headings: bool = True,
                 merge_empty_lines: bool = True, generate_toc: bool = False,
                 validate_latex: bool = True, check_links: bool = False,
                 max_empty_lines: int = 2):
        """
        Initialize post-processor

        Args:
            fix_tables: Fix table alignment
            fix_headings: Standardize heading levels
            merge_empty_lines: Merge consecutive empty lines
            generate_toc: Generate table of contents
            validate_latex: Validate LaTeX formulas
            check_links: Check link validity
            max_empty_lines: Maximum consecutive empty lines
        """
        self.fix_tables = fix_tables
        self.fix_headings = fix_headings
        self.merge_empty_lines = merge_empty_lines
        self.generate_toc = generate_toc
        self.validate_latex = validate_latex
        self.check_links = check_links
        self.max_empty_lines = max_empty_lines

    def process_file(self, file_path: Path) -> PostProcessResult:
        """Process a single Markdown file"""
        result = PostProcessResult(file_path=str(file_path), original_size=0, processed_size=0)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result.original_size = len(content)
        except Exception as e:
            result.errors.append(f"Unable to read file: {e}")
            return result

        # Apply various fixes
        # First fix LLM output issues
        content, json_fixes = self._remove_json_blocks(content)
        if json_fixes > 0:
            result.fixes_applied.append(f"Removed {json_fixes} JSON blocks")

        content, newline_fixes = self._fix_literal_newlines(content)
        if newline_fixes > 0:
            result.fixes_applied.append(f"Fixed {newline_fixes} literal newlines")

        if self.merge_empty_lines:
            content, count = self._merge_empty_lines(content)
            if count > 0:
                result.fixes_applied.append(f"Merged {count} consecutive empty lines")

        if self.fix_tables:
            content, count = self._fix_table_alignment(content)
            if count > 0:
                result.fixes_applied.append(f"Fixed {count} table alignments")

        if self.fix_headings:
            content, fixes = self._standardize_headings(content)
            if fixes:
                result.fixes_applied.extend(fixes)

        if self.validate_latex:
            warnings = self._validate_latex_formulas(content)
            result.warnings.extend(warnings)

        if self.check_links:
            warnings = self._check_broken_links(content)
            result.warnings.extend(warnings)

        # Clean trailing whitespace
        content = self._clean_trailing_whitespace(content)

        # Generate TOC (at the beginning of the file)
        if self.generate_toc:
            toc, has_headings = self._generate_toc(content)
            if has_headings:
                content = self._insert_toc(content, toc)
                result.toc_generated = True
                result.fixes_applied.append("Generated table of contents")

        result.processed_size = len(content)

        # Write back to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            result.errors.append(f"Unable to write file: {e}")

        return result

    def _remove_json_blocks(self, content: str) -> Tuple[str, int]:
        """
        Remove raw JSON blocks from LLM output

        Sometimes LLM outputs raw JSON response instead of extracted text content, like:
        ```json
        {
          "body_text": "actual content...",
          "tables": [...],
          ...
        }
        ```

        This method tries to extract body_text field content, or remove the entire JSON block
        Also handles unclosed JSON blocks
        """
        import json as json_module

        fixes = 0

        # First handle complete ```json ... ``` blocks
        json_block_pattern = r'```json\s*\n(.*?)\n```'

        def extract_or_remove(match):
            nonlocal fixes
            json_str = match.group(1)

            try:
                # Try to parse JSON
                data = json_module.loads(json_str)

                # Try to extract body_text
                if isinstance(data, dict):
                    if 'body_text' in data:
                        fixes += 1
                        body = data['body_text']
                        # If body_text is a string, return directly
                        if isinstance(body, str):
                            return body.replace('\\n', '\n')
                        return str(body)
                    elif 'content' in data:
                        fixes += 1
                        return str(data['content']).replace('\\n', '\n')
                    elif 'text' in data:
                        fixes += 1
                        return str(data['text']).replace('\\n', '\n')

                # Unable to extract useful content, remove entire block
                fixes += 1
                return ''

            except json_module.JSONDecodeError:
                # Not valid JSON, keep original
                return match.group(0)

        content = re.sub(json_block_pattern, extract_or_remove, content, flags=re.DOTALL)

        # Then handle unclosed ```json blocks (until next ## Page or end of file)
        # These blocks typically have the format:
        # ```json
        # {
        #   "page_number": X,
        #   "tables": [...],
        #   "figures": [...],
        #   "formulas": [...],
        #   "body_text": "content..."
        # (not closed)

        # Pattern 1: ```json block until before next ## Page
        unclosed_pattern = r'```json\s*\n\{\s*\n\s*"page_number":\s*\d+,.*?(?=\n\n## Page|\Z)'

        def remove_unclosed_json(match):
            nonlocal fixes
            block = match.group(0)

            # Try to extract body_text content
            body_match = re.search(r'"body_text":\s*"(.*?)(?:"\s*,|\"\s*\}|$)', block, re.DOTALL)
            if body_match:
                body_text = body_match.group(1)
                # Handle escapes
                body_text = body_text.replace('\\n', '\n')
                body_text = body_text.replace('\\t', '\t')
                body_text = body_text.replace('\\"', '"')
                fixes += 1
                return body_text

            # Unable to extract, delete entire block
            fixes += 1
            return ''

        content = re.sub(unclosed_pattern, remove_unclosed_json, content, flags=re.DOTALL)

        # Pattern 2: Delete residual JSON opening marks (only opening, no content)
        content = re.sub(r'```json\s*\n\s*\{\s*\n\s*$', '', content, flags=re.MULTILINE)

        # Pattern 3: Clean up residual JSON property lines (like "tables": [], etc.)
        json_property_pattern = r'^\s*"(?:tables|figures|formulas|page_number)":\s*[\[\d].*?,?\s*$'
        lines = content.split('\n')
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # If it's a JSON property line, skip
            if re.match(json_property_pattern, line):
                fixes += 1
                i += 1
                continue
            # If it starts with "body_text":, extract content
            body_match = re.match(r'^\s*"body_text":\s*"(.*)$', line)
            if body_match:
                # Extract until next ## Page or end
                body_content = [body_match.group(1)]
                i += 1
                while i < len(lines):
                    if lines[i].startswith('## Page') or lines[i].startswith('```'):
                        break
                    # If line ends with ",, remove it
                    line_content = lines[i]
                    if line_content.endswith('",'):
                        line_content = line_content[:-2]
                    body_content.append(line_content)
                    i += 1
                # Merge content
                merged = '\n'.join(body_content)
                merged = merged.replace('\\n', '\n')
                merged = merged.replace('\\t', '\t')
                merged = merged.replace('\\"', '"')
                # Remove trailing quotes and commas
                merged = re.sub(r'[",]+\s*$', '', merged)
                cleaned_lines.append(merged)
                fixes += 1
                continue
            cleaned_lines.append(line)
            i += 1

        content = '\n'.join(cleaned_lines)

        return content, fixes

    def _fix_literal_newlines(self, content: str) -> Tuple[str, int]:
        """
        Fix literal newline characters in LLM output

        Sometimes LLM outputs literal \\n instead of actual newlines
        Example: "line1\\nline2" should become "line1\nline2"

        Note: Be careful not to break:
        - \\n in LaTeX formulas (should be \\newline)
        - Escapes in code blocks
        """
        # Count replacements
        # Match \n not in code blocks or LaTeX

        # Strategy: First protect code blocks and LaTeX blocks, then replace, then restore

        # Protect code blocks
        code_blocks = []
        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f'__CODE_BLOCK_{len(code_blocks) - 1}__'

        # Protect ``` code blocks
        protected = re.sub(r'```.*?```', save_code_block, content, flags=re.DOTALL)

        # Protect LaTeX blocks $$ ... $$
        latex_blocks = []
        def save_latex_block(match):
            latex_blocks.append(match.group(0))
            return f'__LATEX_BLOCK_{len(latex_blocks) - 1}__'

        protected = re.sub(r'\$\$.*?\$\$', save_latex_block, protected, flags=re.DOTALL)

        # Count and replace literal \n
        # Note: In the raw string, \\n represents backslash followed by n
        original_count = protected.count('\\n')
        protected = protected.replace('\\n', '\n')
        fixes = original_count

        # Restore LaTeX blocks
        for i, block in enumerate(latex_blocks):
            protected = protected.replace(f'__LATEX_BLOCK_{i}__', block)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            protected = protected.replace(f'__CODE_BLOCK_{i}__', block)

        return protected, fixes

    def _merge_empty_lines(self, content: str) -> Tuple[str, int]:
        """Merge consecutive empty lines"""
        # Match consecutive empty lines exceeding max_empty_lines
        pattern = r'\n{' + str(self.max_empty_lines + 2) + r',}'
        replacement = '\n' * (self.max_empty_lines + 1)

        # Count replacements
        matches = re.findall(pattern, content)
        count = len(matches)

        content = re.sub(pattern, replacement, content)
        return content, count

    def _fix_table_alignment(self, content: str) -> Tuple[str, int]:
        """Fix table alignment"""
        lines = content.split('\n')
        new_lines = []
        i = 0
        fixed_count = 0

        while i < len(lines):
            line = lines[i]

            # Detect table start (lines containing |)
            if '|' in line and i + 1 < len(lines) and re.match(r'^\s*\|[-:|]+\|', lines[i + 1]):
                # Find table range
                table_start = i
                table_end = i

                # Find table end
                while table_end < len(lines) and '|' in lines[table_end]:
                    table_end += 1

                # Extract table rows
                table_lines = lines[table_start:table_end]

                # Fix table
                fixed_table = self._align_table(table_lines)
                if fixed_table != table_lines:
                    fixed_count += 1

                new_lines.extend(fixed_table)
                i = table_end
            else:
                new_lines.append(line)
                i += 1

        return '\n'.join(new_lines), fixed_count

    def _align_table(self, table_lines: List[str]) -> List[str]:
        """Align a single table"""
        if len(table_lines) < 2:
            return table_lines

        # Parse cells in each row
        rows = []
        for line in table_lines:
            # Remove leading/trailing |
            line = line.strip()
            if line.startswith('|'):
                line = line[1:]
            if line.endswith('|'):
                line = line[:-1]

            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)

        if not rows:
            return table_lines

        # Calculate max width for each column
        num_cols = max(len(row) for row in rows)
        col_widths = [0] * num_cols

        for row in rows:
            for j, cell in enumerate(row):
                if j < num_cols:
                    # Don't calculate width for separator rows
                    if not re.match(r'^[-:]+$', cell):
                        col_widths[j] = max(col_widths[j], len(cell))

        # Minimum width is 3 (separator needs at least 3 chars ---)
        col_widths = [max(w, 3) for w in col_widths]

        # Rebuild table
        aligned_lines = []
        for i, row in enumerate(rows):
            cells = []
            for j in range(num_cols):
                cell = row[j] if j < len(row) else ''

                # Special handling for separator row
                if i == 1 and re.match(r'^[-:]+$', cell):
                    # Preserve alignment markers
                    if cell.startswith(':') and cell.endswith(':'):
                        cell = ':' + '-' * (col_widths[j] - 2) + ':'
                    elif cell.startswith(':'):
                        cell = ':' + '-' * (col_widths[j] - 1)
                    elif cell.endswith(':'):
                        cell = '-' * (col_widths[j] - 1) + ':'
                    else:
                        cell = '-' * col_widths[j]
                else:
                    cell = cell.ljust(col_widths[j])

                cells.append(cell)

            aligned_lines.append('| ' + ' | '.join(cells) + ' |')

        return aligned_lines

    def _standardize_headings(self, content: str) -> Tuple[str, List[str]]:
        """Standardize heading levels"""
        fixes = []
        lines = content.split('\n')
        new_lines = []

        # Detect orphan headings (heading directly followed by another heading)
        for i, line in enumerate(lines):
            new_lines.append(line)

            # Detect heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)

                # Check if next line is also a heading (orphan heading)
                if i + 1 < len(lines):
                    next_match = re.match(r'^(#{1,6})\s+', lines[i + 1])
                    if next_match:
                        next_level = len(next_match.group(1))
                        # If next heading level is lower (larger number) with no content in between, warn
                        if next_level > level:
                            fixes.append(f"Warning: Line {i+1} heading '{title[:30]}...' has no content after it")

        # Check heading level jumps
        prev_level = 0
        for i, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,6})\s+', line)
            if heading_match:
                level = len(heading_match.group(1))
                if prev_level > 0 and level > prev_level + 1:
                    fixes.append(f"Warning: Line {i+1} heading level jump (H{prev_level} → H{level})")
                prev_level = level

        return '\n'.join(new_lines), fixes

    def _validate_latex_formulas(self, content: str) -> List[str]:
        """Validate LaTeX formula syntax"""
        warnings = []

        # Detect inline formulas $...$
        inline_formulas = re.findall(r'\$([^$]+)\$', content)
        for formula in inline_formulas:
            issues = self._check_latex_syntax(formula)
            if issues:
                warnings.append(f"Inline formula issue: ${formula[:30]}... - {issues}")

        # Detect block formulas $$...$$
        block_formulas = re.findall(r'\$\$([^$]+)\$\$', content, re.DOTALL)
        for formula in block_formulas:
            issues = self._check_latex_syntax(formula)
            if issues:
                warnings.append(f"Block formula issue: $${formula[:30]}... - {issues}")

        return warnings

    def _check_latex_syntax(self, formula: str) -> Optional[str]:
        """Check single LaTeX formula syntax"""
        # Check bracket matching
        brackets = {'{': '}', '(': ')', '[': ']'}
        stack = []

        for char in formula:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return "Brackets mismatch"
                expected = brackets[stack.pop()]
                if char != expected:
                    return "Brackets mismatch"

        if stack:
            return "Unclosed brackets"

        # Check if common commands are complete
        incomplete_commands = [
            (r'\\frac\s*$', '\\frac missing arguments'),
            (r'\\sqrt\s*$', '\\sqrt missing arguments'),
            (r'\\sum\s*$', '\\sum may be missing subscript/superscript'),
        ]

        for pattern, msg in incomplete_commands:
            if re.search(pattern, formula):
                return msg

        return None

    def _check_broken_links(self, content: str) -> List[str]:
        """Check for potentially broken links"""
        warnings = []

        # Detect Markdown links [text](url)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, url in links:
            if url.startswith('#'):
                # Internal anchor link, check if it exists
                anchor = url[1:]
                # Simplified check: see if there's a matching heading
                heading_pattern = re.escape(anchor.replace('-', ' '))
                if not re.search(rf'#{1,6}\s+{heading_pattern}', content, re.IGNORECASE):
                    warnings.append(f"Internal link may be broken: [{text}]({url})")
            elif not url.startswith(('http://', 'https://', 'mailto:', '/')):
                # Relative path link
                warnings.append(f"Relative path link (needs verification): [{text}]({url})")

        return warnings

    def _clean_trailing_whitespace(self, content: str) -> str:
        """Clean trailing whitespace"""
        lines = content.split('\n')
        cleaned = [line.rstrip() for line in lines]
        return '\n'.join(cleaned)

    def _generate_toc(self, content: str) -> Tuple[str, bool]:
        """Generate table of contents"""
        headings = []

        # Extract all headings
        for line in content.split('\n'):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                # Generate anchor
                anchor = re.sub(r'[^\w\s-]', '', title.lower())
                anchor = re.sub(r'\s+', '-', anchor)
                headings.append((level, title, anchor))

        if not headings:
            return '', False

        # Generate TOC
        toc_lines = ['## Table of Contents\n']
        for level, title, anchor in headings:
            if level == 1:
                continue  # Skip level 1 headings (usually document title)
            indent = '  ' * (level - 2)
            toc_lines.append(f'{indent}- [{title}](#{anchor})')

        toc_lines.append('\n---\n')
        return '\n'.join(toc_lines), True

    def _insert_toc(self, content: str, toc: str) -> str:
        """Insert table of contents at appropriate position"""
        lines = content.split('\n')

        # Find and insert after the first level 1 heading
        for i, line in enumerate(lines):
            if re.match(r'^#\s+', line):
                # Insert after level 1 heading
                lines.insert(i + 1, '')
                lines.insert(i + 2, toc)
                break
        else:
            # No level 1 heading, insert at the beginning
            lines.insert(0, toc)

        return '\n'.join(lines)


def process_directory(input_dir: Path, **kwargs) -> List[PostProcessResult]:
    """Process all Markdown files in a directory"""
    processor = MarkdownPostProcessor(**kwargs)
    results = []

    # Find all .md files
    md_files = list(input_dir.rglob('*.md'))

    print(f"📝 Found {len(md_files)} Markdown files")

    for md_file in sorted(md_files):
        # Skip quality report files
        if md_file.name == 'QUALITY_REPORT.md':
            continue

        print(f"   Processing: {md_file.name}...", end=' ')
        result = processor.process_file(md_file)

        if result.errors:
            print(f"❌ Error: {result.errors[0]}")
        elif result.fixes_applied:
            print(f"✅ {len(result.fixes_applied)} fixes applied")
        else:
            print("✓ No fixes needed")

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='DocMind Markdown Post-processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./final-delivery
  %(prog)s --input ./final-delivery --generate-toc
  %(prog)s --input ./final-delivery --fix-tables --fix-headings
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory (containing Markdown files)'
    )

    parser.add_argument(
        '--fix-tables',
        action='store_true',
        default=True,
        help='Fix table alignment (default: enabled)'
    )

    parser.add_argument(
        '--no-fix-tables',
        action='store_true',
        help='Disable table alignment fix'
    )

    parser.add_argument(
        '--fix-headings',
        action='store_true',
        default=True,
        help='Standardize heading levels (default: enabled)'
    )

    parser.add_argument(
        '--no-fix-headings',
        action='store_true',
        help='Disable heading fix'
    )

    parser.add_argument(
        '--merge-empty-lines',
        action='store_true',
        default=True,
        help='Merge consecutive empty lines (default: enabled)'
    )

    parser.add_argument(
        '--generate-toc',
        action='store_true',
        default=False,
        help='Generate table of contents (default: disabled)'
    )

    parser.add_argument(
        '--validate-latex',
        action='store_true',
        default=True,
        help='Validate LaTeX formulas (default: enabled)'
    )

    parser.add_argument(
        '--check-links',
        action='store_true',
        default=False,
        help='Check link validity (default: disabled)'
    )

    parser.add_argument(
        '--max-empty-lines',
        type=int,
        default=2,
        help='Maximum consecutive empty lines (default: 2)'
    )

    parser.add_argument(
        '--report',
        help='Output report file path (optional)'
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Process arguments
    fix_tables = args.fix_tables and not args.no_fix_tables
    fix_headings = args.fix_headings and not args.no_fix_headings

    print("=" * 60)
    print("📝 Markdown Post-processor")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Fix tables: {'Yes' if fix_tables else 'No'}")
    print(f"Fix headings: {'Yes' if fix_headings else 'No'}")
    print(f"Merge empty lines: {'Yes' if args.merge_empty_lines else 'No'}")
    print(f"Generate TOC: {'Yes' if args.generate_toc else 'No'}")
    print(f"Validate LaTeX: {'Yes' if args.validate_latex else 'No'}")
    print("=" * 60)
    print()

    # 处理文件
    results = process_directory(
        input_dir,
        fix_tables=fix_tables,
        fix_headings=fix_headings,
        merge_empty_lines=args.merge_empty_lines,
        generate_toc=args.generate_toc,
        validate_latex=args.validate_latex,
        check_links=args.check_links,
        max_empty_lines=args.max_empty_lines
    )

    # Aggregate results
    total_fixes = sum(len(r.fixes_applied) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    files_with_toc = sum(1 for r in results if r.toc_generated)

    print()
    print("=" * 60)
    print("📊 Processing Results")
    print("=" * 60)
    print(f"Files processed: {len(results)}")
    print(f"Total fixes: {total_fixes}")
    print(f"TOC generated: {files_with_toc}")
    print(f"Warnings: {total_warnings}")
    print(f"Errors: {total_errors}")

    # Display warnings
    if total_warnings > 0:
        print()
        print("⚠️  Warnings:")
        for r in results:
            for w in r.warnings[:5]:  # Show at most 5
                print(f"   {Path(r.file_path).name}: {w}")
            if len(r.warnings) > 5:
                print(f"   ... and {len(r.warnings) - 5} more warnings")

    # Display errors
    if total_errors > 0:
        print()
        print("❌ Errors:")
        for r in results:
            for e in r.errors:
                print(f"   {Path(r.file_path).name}: {e}")

    print("=" * 60)

    # Optional: Generate report file
    if args.report:
        report_path = Path(args.report)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Markdown Post-processing Report\n\n")
            f.write(f"| File | Original Size | After Processing | Fixes | Warnings |\n")
            f.write("|------|---------------|------------------|-------|----------|\n")
            for r in results:
                name = Path(r.file_path).name[:30]
                f.write(f"| {name} | {r.original_size:,} | {r.processed_size:,} | {len(r.fixes_applied)} | {len(r.warnings)} |\n")
        print(f"📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()
