#!/usr/bin/env python3
"""
Progress Manager - Support checkpoint resume
Manages PDF processing progress state, supports resumption after interruption
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from filelock import FileLock
import threading

class ProgressManager:
    """Progress Manager"""

    VERSION = "1.0"

    def __init__(self, progress_file: str = None, base_dir: str = None):
        """
        Initialize progress manager

        Args:
            progress_file: Progress file path, default is base_dir/progress.json
            base_dir: Base directory, default is parent directory of the script
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)

        if progress_file is None:
            self.progress_file = self.base_dir / "progress.json"
        else:
            self.progress_file = Path(progress_file)

        self.lock_file = str(self.progress_file) + ".lock"
        self._lock = threading.Lock()
        self._data = None

    def _get_file_lock(self):
        """Get file lock"""
        return FileLock(self.lock_file, timeout=10)

    def load(self) -> Dict:
        """Load progress file"""
        with self._lock:
            if self.progress_file.exists():
                try:
                    with self._get_file_lock():
                        with open(self.progress_file, 'r', encoding='utf-8') as f:
                            self._data = json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"⚠️  Progress file corrupted, creating new file: {e}")
                    self._data = self._create_new_progress()
            else:
                self._data = self._create_new_progress()
            return self._data

    def save(self):
        """Save progress file"""
        with self._lock:
            if self._data is None:
                return

            self._data['updated_at'] = datetime.now().isoformat()

            with self._get_file_lock():
                # Write to temp file first, then rename (atomic operation)
                temp_file = str(self.progress_file) + ".tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, self.progress_file)

    def _create_new_progress(self) -> Dict:
        """Create new progress structure"""
        return {
            "version": self.VERSION,
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "initialized",
            "steps": {
                "split": {"status": "pending"},
                "process_chunks": {"status": "pending", "completed": [], "failed": [], "pending": []},
                "process_direct": {"status": "pending", "completed": [], "failed": [], "pending": []},
                "merge": {"status": "pending"}
            },
            "pdf_progress": {},
            "statistics": {
                "total_pdfs": 0,
                "completed_pdfs": 0,
                "total_pages": 0,
                "completed_pages": 0,
                "failed_pages": 0
            }
        }

    def reset(self):
        """Reset progress (start from beginning)"""
        self._data = self._create_new_progress()
        self.save()
        print("✅ Progress reset")

    # ============ Step Management ============

    def get_step_status(self, step_name: str) -> str:
        """Get step status"""
        if self._data is None:
            self.load()
        return self._data.get('steps', {}).get(step_name, {}).get('status', 'pending')

    def set_step_status(self, step_name: str, status: str):
        """Set step status"""
        if self._data is None:
            self.load()

        if step_name not in self._data['steps']:
            self._data['steps'][step_name] = {}

        self._data['steps'][step_name]['status'] = status

        if status == 'completed':
            self._data['steps'][step_name]['completed_at'] = datetime.now().isoformat()
        elif status == 'in_progress':
            self._data['steps'][step_name]['started_at'] = datetime.now().isoformat()

        self.save()

    def is_step_completed(self, step_name: str) -> bool:
        """Check if step is completed"""
        return self.get_step_status(step_name) == 'completed'

    # ============ PDF Level Management ============

    def get_pending_pdfs(self, step_name: str) -> List[str]:
        """Get list of pending PDFs"""
        if self._data is None:
            self.load()
        return self._data.get('steps', {}).get(step_name, {}).get('pending', [])

    def get_completed_pdfs(self, step_name: str) -> List[str]:
        """Get list of completed PDFs"""
        if self._data is None:
            self.load()
        return self._data.get('steps', {}).get(step_name, {}).get('completed', [])

    def get_failed_pdfs(self, step_name: str) -> List[str]:
        """Get list of failed PDFs"""
        if self._data is None:
            self.load()
        return self._data.get('steps', {}).get(step_name, {}).get('failed', [])

    def set_pdf_list(self, step_name: str, pending: List[str]):
        """Set list of pending PDFs"""
        if self._data is None:
            self.load()

        if step_name not in self._data['steps']:
            self._data['steps'][step_name] = {}

        # Filter out already completed ones
        completed = self._data['steps'][step_name].get('completed', [])
        pending = [p for p in pending if p not in completed]

        self._data['steps'][step_name]['pending'] = pending
        if 'completed' not in self._data['steps'][step_name]:
            self._data['steps'][step_name]['completed'] = []
        if 'failed' not in self._data['steps'][step_name]:
            self._data['steps'][step_name]['failed'] = []

        self.save()

    def mark_pdf_completed(self, step_name: str, pdf_name: str):
        """Mark PDF as completed"""
        if self._data is None:
            self.load()

        step = self._data['steps'].get(step_name, {})

        # Remove from pending and failed
        if pdf_name in step.get('pending', []):
            step['pending'].remove(pdf_name)
        if pdf_name in step.get('failed', []):
            step['failed'].remove(pdf_name)

        # Add to completed
        if pdf_name not in step.get('completed', []):
            if 'completed' not in step:
                step['completed'] = []
            step['completed'].append(pdf_name)

        self._data['steps'][step_name] = step
        self._data['statistics']['completed_pdfs'] = len(step.get('completed', []))
        self.save()

    def mark_pdf_failed(self, step_name: str, pdf_name: str, error: str = None):
        """Mark PDF as failed"""
        if self._data is None:
            self.load()

        step = self._data['steps'].get(step_name, {})

        # Remove from pending
        if pdf_name in step.get('pending', []):
            step['pending'].remove(pdf_name)

        # Add to failed
        if pdf_name not in step.get('failed', []):
            if 'failed' not in step:
                step['failed'] = []
            step['failed'].append(pdf_name)

        # Record error message
        if error:
            if 'errors' not in step:
                step['errors'] = {}
            step['errors'][pdf_name] = {
                'error': error,
                'timestamp': datetime.now().isoformat()
            }

        self._data['steps'][step_name] = step
        self.save()

    def is_pdf_completed(self, step_name: str, pdf_name: str) -> bool:
        """Check if PDF is completed (only checks list)"""
        return pdf_name in self.get_completed_pdfs(step_name)

    def validate_pdf_completion(self, step_name: str, pdf_name: str, output_dir: str,
                                 min_content_size: int = 500,
                                 min_completion_rate: float = 0.90) -> dict:
        """
        Validate if PDF is truly completed (check actual output)

        Args:
            step_name: Step name
            pdf_name: PDF filename (without extension)
            output_dir: Output directory path
            min_content_size: Minimum MD file content size (bytes)
            min_completion_rate: Minimum page completion rate (0.0-1.0)

        Returns:
            dict: {
                'valid': bool,           # Is validly completed
                'reason': str,           # Reason description
                'md_exists': bool,       # Does MD file exist
                'md_size': int,          # MD file size
                'page_completion_rate': float,  # Page completion rate
                'should_reprocess': bool # Should be reprocessed
            }
        """
        from pathlib import Path

        result = {
            'valid': False,
            'reason': '',
            'md_exists': False,
            'md_size': 0,
            'page_completion_rate': 0.0,
            'should_reprocess': False
        }

        # Check if in completed list
        if not self.is_pdf_completed(step_name, pdf_name):
            result['reason'] = 'Not in completed list'
            result['should_reprocess'] = True
            return result

        # Check output directory and MD file
        output_path = Path(output_dir) / pdf_name
        md_files = list(output_path.glob("*.md")) if output_path.exists() else []

        if not md_files:
            result['reason'] = 'MD file does not exist'
            result['should_reprocess'] = True
            return result

        # Check MD file size
        md_file = md_files[0]
        md_size = md_file.stat().st_size
        result['md_exists'] = True
        result['md_size'] = md_size

        if md_size < min_content_size:
            result['reason'] = f'MD file too small ({md_size} bytes < {min_content_size})'
            result['should_reprocess'] = True
            return result

        # Check page completion rate
        if self._data is None:
            self.load()

        pdf_progress = self._data.get('pdf_progress', {}).get(pdf_name, {})
        total_pages = pdf_progress.get('total_pages', 0)
        completed_pages = len(pdf_progress.get('completed_pages', []))

        if total_pages > 0:
            completion_rate = completed_pages / total_pages
            result['page_completion_rate'] = completion_rate

            if completion_rate < min_completion_rate:
                result['reason'] = f'Page completion rate insufficient ({completion_rate:.1%} < {min_completion_rate:.0%})'
                result['should_reprocess'] = True
                return result

        # All checks passed
        result['valid'] = True
        result['reason'] = 'Validation passed'
        return result

    def invalidate_pdf_completion(self, step_name: str, pdf_name: str):
        """
        Move PDF from completed back to pending (for reprocessing invalid "completed" status)
        """
        if self._data is None:
            self.load()

        step = self._data['steps'].get(step_name, {})

        # Remove from completed
        if pdf_name in step.get('completed', []):
            step['completed'].remove(pdf_name)
            print(f"   ⚠️  Removed from completed: {pdf_name}")

        # Add back to pending
        if pdf_name not in step.get('pending', []):
            if 'pending' not in step:
                step['pending'] = []
            step['pending'].append(pdf_name)

        # Clear page progress (allow reprocessing)
        if pdf_name in self._data.get('pdf_progress', {}):
            old_progress = self._data['pdf_progress'][pdf_name]
            # Update statistics
            self._data['statistics']['completed_pages'] -= len(old_progress.get('completed_pages', []))
            self._data['statistics']['failed_pages'] -= len(old_progress.get('failed_pages', []))
            # Delete page progress
            del self._data['pdf_progress'][pdf_name]

        self._data['steps'][step_name] = step
        self.save()

    # ============ Page Level Management ============

    def init_pdf_progress(self, pdf_name: str, total_pages: int):
        """Initialize PDF page progress"""
        if self._data is None:
            self.load()

        if pdf_name not in self._data['pdf_progress']:
            self._data['pdf_progress'][pdf_name] = {
                'total_pages': total_pages,
                'completed_pages': [],
                'failed_pages': [],
                'started_at': datetime.now().isoformat()
            }
            self._data['statistics']['total_pages'] += total_pages
            self.save()

    def get_completed_pages(self, pdf_name: str) -> List[int]:
        """Get list of completed pages"""
        if self._data is None:
            self.load()
        return self._data.get('pdf_progress', {}).get(pdf_name, {}).get('completed_pages', [])

    def get_failed_pages(self, pdf_name: str) -> List[int]:
        """Get list of failed pages"""
        if self._data is None:
            self.load()
        return self._data.get('pdf_progress', {}).get(pdf_name, {}).get('failed_pages', [])

    def mark_page_completed(self, pdf_name: str, page_num: int):
        """Mark page as completed"""
        if self._data is None:
            self.load()

        if pdf_name not in self._data['pdf_progress']:
            self._data['pdf_progress'][pdf_name] = {
                'total_pages': 0,
                'completed_pages': [],
                'failed_pages': []
            }

        progress = self._data['pdf_progress'][pdf_name]

        # Remove from failed
        if page_num in progress.get('failed_pages', []):
            progress['failed_pages'].remove(page_num)

        # Add to completed
        if page_num not in progress.get('completed_pages', []):
            progress['completed_pages'].append(page_num)
            progress['completed_pages'].sort()
            self._data['statistics']['completed_pages'] += 1

        self.save()

    def mark_page_failed(self, pdf_name: str, page_num: int, error: str = None):
        """Mark page as failed"""
        if self._data is None:
            self.load()

        if pdf_name not in self._data['pdf_progress']:
            self._data['pdf_progress'][pdf_name] = {
                'total_pages': 0,
                'completed_pages': [],
                'failed_pages': []
            }

        progress = self._data['pdf_progress'][pdf_name]

        if page_num not in progress.get('failed_pages', []):
            progress['failed_pages'].append(page_num)
            progress['failed_pages'].sort()
            self._data['statistics']['failed_pages'] += 1

        if error:
            if 'page_errors' not in progress:
                progress['page_errors'] = {}
            progress['page_errors'][str(page_num)] = error

        self.save()

    def is_page_completed(self, pdf_name: str, page_num: int) -> bool:
        """Check if page is completed"""
        return page_num in self.get_completed_pages(pdf_name)

    def get_pdf_progress_percent(self, pdf_name: str) -> float:
        """Get PDF processing progress percentage"""
        if self._data is None:
            self.load()

        progress = self._data.get('pdf_progress', {}).get(pdf_name, {})
        total = progress.get('total_pages', 0)
        completed = len(progress.get('completed_pages', []))

        if total == 0:
            return 0.0
        return (completed / total) * 100

    # ============ Status Queries ============

    def get_status_summary(self) -> Dict:
        """Get status summary"""
        if self._data is None:
            self.load()

        return {
            'status': self._data.get('status', 'unknown'),
            'started_at': self._data.get('started_at'),
            'updated_at': self._data.get('updated_at'),
            'steps': {
                name: {
                    'status': step.get('status', 'pending'),
                    'completed': len(step.get('completed', [])),
                    'pending': len(step.get('pending', [])),
                    'failed': len(step.get('failed', []))
                }
                for name, step in self._data.get('steps', {}).items()
            },
            'statistics': self._data.get('statistics', {})
        }

    def print_status(self):
        """Print status report"""
        summary = self.get_status_summary()

        print("\n" + "=" * 60)
        print("📊 DocMind Processing Progress")
        print("=" * 60)
        print(f"Status: {summary['status']}")
        print(f"Started: {summary['started_at']}")
        print(f"Updated: {summary['updated_at']}")
        print()

        print("Step Status:")
        for step_name, step_info in summary['steps'].items():
            status_icon = {
                'completed': '✅',
                'in_progress': '🔄',
                'pending': '⏳',
                'failed': '❌'
            }.get(step_info['status'], '❓')

            print(f"  {status_icon} {step_name}: {step_info['status']}")
            if step_info['completed'] > 0 or step_info['pending'] > 0:
                print(f"      Completed: {step_info['completed']}, Pending: {step_info['pending']}, Failed: {step_info['failed']}")

        print()
        stats = summary['statistics']
        print("Statistics:")
        print(f"  PDFs: {stats.get('completed_pdfs', 0)}/{stats.get('total_pdfs', 0)}")
        print(f"  Pages: {stats.get('completed_pages', 0)}/{stats.get('total_pages', 0)} (Failed: {stats.get('failed_pages', 0)})")

        if stats.get('total_pages', 0) > 0:
            progress = (stats.get('completed_pages', 0) / stats['total_pages']) * 100
            print(f"  Overall progress: {progress:.1f}%")

        print("=" * 60)

    def set_overall_status(self, status: str):
        """Set overall status"""
        if self._data is None:
            self.load()
        self._data['status'] = status
        self.save()

    def is_all_completed(self) -> bool:
        """Check if all completed"""
        if self._data is None:
            self.load()

        for step_name, step in self._data.get('steps', {}).items():
            if step.get('status') != 'completed':
                return False
        return True


# Convenience functions
_default_manager = None

def get_progress_manager(progress_file: str = None, base_dir: str = None) -> ProgressManager:
    """Get progress manager singleton"""
    global _default_manager
    if _default_manager is None or progress_file is not None:
        _default_manager = ProgressManager(progress_file, base_dir)
    return _default_manager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Progress Manager")
    parser.add_argument("--status", action="store_true", help="Show current progress")
    parser.add_argument("--reset", action="store_true", help="Reset progress")
    parser.add_argument("--progress-file", default=None, help="Progress file path")

    args = parser.parse_args()

    manager = ProgressManager(args.progress_file)
    manager.load()

    if args.reset:
        manager.reset()
    elif args.status:
        manager.print_status()
    else:
        manager.print_status()
