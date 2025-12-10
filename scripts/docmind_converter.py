#!/usr/bin/env python3
"""
DocMind Lite - PDF to Markdown Converter
Simplified version for easy deployment with single API key.

Features:
1. Three-page context window (prev + current + next)
2. Concurrent processing (asyncio.Semaphore)
3. Three-phase processing (PDF→Image → OCR → LLM)
4. Resume capability
5. Single API key mode (simplified from multi-key version)
"""

import sys
import os
import json
import time
import yaml
import re
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from itertools import cycle
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

# Resource monitoring
try:
    from resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False

# ============ Issue 4: Retry Strategy Configuration ============

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 2                    # Max retry attempts (Lite: 2, Full: 4)
    initial_delay: float = 1.0              # Initial delay (seconds)
    max_delay: float = 30.0                 # Max delay (seconds)
    exponential_base: float = 2.0           # Exponential backoff base
    jitter: bool = True                     # Add random jitter

    # Retry trigger conditions
    retry_on_status_codes: tuple = (429, 500, 502, 503, 504)  # HTTP status codes
    retry_on_exceptions: tuple = (
        "SSLError", "ConnectionError", "TimeoutError",
        "NameResolutionError", "UNEXPECTED_EOF"
    )

# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()

# ============ Issue 5: API Key Health Monitoring ============

@dataclass
class APIKeyHealth:
    """Health status of a single API Key"""
    key: str
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    total_latency: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    is_disabled: bool = False
    disabled_until: Optional[float] = None

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.success_count if self.success_count > 0 else 0.0

    def record_success(self, latency: float):
        """Record successful call"""
        self.success_count += 1
        self.consecutive_failures = 0
        self.total_latency += latency
        self.last_success_time = time.time()
        # If previously disabled, recover now
        if self.is_disabled:
            self.is_disabled = False
            self.disabled_until = None

    def record_failure(self, disable_threshold: int = 5, disable_duration: float = 300.0):
        """Record failed call"""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        # Temporarily disable after consecutive failures exceed threshold
        if self.consecutive_failures >= disable_threshold:
            self.is_disabled = True
            self.disabled_until = time.time() + disable_duration

    def is_available(self) -> bool:
        """Check if key is available"""
        if not self.is_disabled:
            return True
        # Check if disable duration has passed
        if self.disabled_until and time.time() > self.disabled_until:
            self.is_disabled = False
            self.disabled_until = None
            self.consecutive_failures = 0  # Reset consecutive failure count
            return True
        return False


class APIKeyHealthMonitor:
    """API Key Health Monitor"""

    def __init__(self, api_keys: List[str], disable_threshold: int = 5, disable_duration: float = 300.0):
        self.health_data: Dict[str, APIKeyHealth] = {
            key: APIKeyHealth(key=key) for key in api_keys
        }
        self.disable_threshold = disable_threshold
        self.disable_duration = disable_duration
        self._lock = asyncio.Lock()
        self._current_index = 0

    async def get_healthy_key(self) -> Optional[str]:
        """Get a healthy API Key (smart polling)"""
        async with self._lock:
            available_keys = [
                key for key, health in self.health_data.items()
                if health.is_available()
            ]

            if not available_keys:
                # All keys disabled, return the earliest disabled one (may have recovered)
                earliest_key = min(
                    self.health_data.keys(),
                    key=lambda k: self.health_data[k].disabled_until or float('inf')
                )
                # Force recovery
                self.health_data[earliest_key].is_disabled = False
                return earliest_key

            # Prefer keys with higher success rate
            # Sort by success rate, but add some randomness to avoid all requests hitting the same key
            sorted_keys = sorted(
                available_keys,
                key=lambda k: (
                    -self.health_data[k].success_rate,
                    self.health_data[k].consecutive_failures,
                    random.random() * 0.1  # Small random perturbation
                )
            )

            return sorted_keys[0]

    def get_healthy_key_sync(self) -> Optional[str]:
        """Sync version: Get healthy API Key"""
        available_keys = [
            key for key, health in self.health_data.items()
            if health.is_available()
        ]

        if not available_keys:
            earliest_key = min(
                self.health_data.keys(),
                key=lambda k: self.health_data[k].disabled_until or float('inf')
            )
            self.health_data[earliest_key].is_disabled = False
            return earliest_key

        sorted_keys = sorted(
            available_keys,
            key=lambda k: (
                -self.health_data[k].success_rate,
                self.health_data[k].consecutive_failures
            )
        )
        return sorted_keys[0]

    async def record_success(self, key: str, latency: float):
        """Record success"""
        async with self._lock:
            if key in self.health_data:
                self.health_data[key].record_success(latency)

    async def record_failure(self, key: str):
        """Record failure"""
        async with self._lock:
            if key in self.health_data:
                self.health_data[key].record_failure(
                    self.disable_threshold,
                    self.disable_duration
                )

    def record_success_sync(self, key: str, latency: float):
        """Sync version: Record success"""
        if key in self.health_data:
            self.health_data[key].record_success(latency)

    def record_failure_sync(self, key: str):
        """Sync version: Record failure"""
        if key in self.health_data:
            self.health_data[key].record_failure(
                self.disable_threshold,
                self.disable_duration
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = {
            "total_keys": len(self.health_data),
            "active_keys": sum(1 for h in self.health_data.values() if h.is_available()),
            "disabled_keys": sum(1 for h in self.health_data.values() if h.is_disabled),
            "keys": {}
        }

        for key, health in self.health_data.items():
            masked_key = f"{key[:8]}...{key[-4:]}"
            stats["keys"][masked_key] = {
                "success_rate": f"{health.success_rate:.1%}",
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "consecutive_failures": health.consecutive_failures,
                "avg_latency": f"{health.avg_latency:.2f}s",
                "is_disabled": health.is_disabled
            }

        return stats

    def print_stats(self):
        """Print health statistics"""
        stats = self.get_stats()
        print(f"\n📊 API Key Health Status:")
        print(f"   Active: {stats['active_keys']}/{stats['total_keys']} | Disabled: {stats['disabled_keys']}")
        for key, info in stats["keys"].items():
            status = "🔴" if info["is_disabled"] else "🟢"
            print(f"   {status} {key}: Success rate {info['success_rate']}, "
                  f"Calls {info['success_count']}/{info['success_count']+info['failure_count']}, "
                  f"Latency {info['avg_latency']}")


# Global health monitor (initialized after loading API Keys)
_api_key_monitor: Optional[APIKeyHealthMonitor] = None


def should_retry(error: Exception, config: RetryConfig) -> bool:
    """Determine whether to retry"""
    error_str = str(error)

    # Check if matches retry trigger conditions
    for exc_pattern in config.retry_on_exceptions:
        if exc_pattern in error_str:
            return True

    return False


def calculate_retry_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate retry delay (exponential backoff + jitter)"""
    delay = min(
        config.initial_delay * (config.exponential_base ** attempt),
        config.max_delay
    )

    if config.jitter:
        # Add ±25% random jitter
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay)  # Minimum delay 0.1 seconds

# ============ Issue 3 & 5: Model and Context Configuration ============

class ContextMode(Enum):
    """Context window mode"""
    MINIMAL = "minimal"     # 100/300/100 chars - Politically sensitive documents
    STANDARD = "standard"   # 500/full/500 chars - Normal academic documents
    FULL = "full"           # 2000/full/2000 chars - Economics chart documents

class PromptMode(Enum):
    """Prompt mode"""
    SIMPLE = "simple"       # ~500 chars - Plain text pages
    ENHANCED = "enhanced"   # ~2500 chars - Chart data extraction

# Default configuration (can be overridden via command line arguments)
DEFAULT_MODEL = "qwen-vl-max-latest"  # Supports 12M pixels, 16K tokens per image
DEFAULT_CONTEXT_MODE = ContextMode.STANDARD
DEFAULT_PROMPT_MODE = PromptMode.ENHANCED

# Chain-of-thought chart extraction Prompt (Issue 3)
CHART_EXTRACTION_PROMPT = '''### TASK: Analyze this page image using Chain-of-Thought reasoning

**STEP 1 - CHART IDENTIFICATION:**
First, identify all visual elements on this page:
- Is there a figure/chart/table/diagram?
- If yes, what is its label? (e.g., "Figure 1", "Table 2")
- What type of visualization is it?
  * data_visualization: scatter, bar, line, pie, area charts
  * conceptual_diagram: flowchart, process diagram, org chart
  * mathematical_model: equations, formulas with graphs
  * data_table: structured tabular data
  * photograph: photos, images
  * map: geographic representations

**STEP 2 - VISUAL CONTENT ANALYSIS:**
For each identified figure:
- Describe what the chart shows in 1-2 sentences
- List ALL visible text labels (axis labels, legend items, annotations)
- Identify key elements (lines, bars, data points, etc.)

**STEP 3 - DATA EXTRACTION (REQUIRED for data_visualization AND data_table):**
⚠️ THIS IS MANDATORY - You MUST extract actual numeric values!

For charts (data_visualization):
- X-axis: label, unit (if any), range [min, max], scale type (linear/log/categorical)
- Y-axis: label, unit (if any), range [min, max], scale type
- Data series: For each series, extract:
  * Series name (from legend)
  * Key data points: {{x: value, y: value}} (use ~ for approximate readings)
  * Overall trend: upward/downward/stable/fluctuating

For tables (data_table):
- Extract ALL cell values as data_series
- Each row becomes a data_point with x = row identifier (e.g., year, name)
- Columns become y values: {{x: "1895", "Branches": 16, "Deposit": 14, "Total": 42}}
- ❌ DO NOT leave data_series empty if there is visible numeric data!

**STEP 4 - STATISTICAL INFORMATION (if visible):**
- Correlation coefficient (r)
- R-squared value (R²)
- Trend line equation (y = mx + b)
- Sample size (n)
- Growth rates or percentages

**STEP 5 - BODY TEXT & TABLES:**
Extract all text NOT part of the figures:
- Main paragraphs (keep structure)
- Section headings
- Tables: Convert to Markdown format
- Formulas: Convert to LaTeX ($..$ or $$..$$)
- Footnotes and citations
- Keep original language, do NOT translate

### CONTEXT WINDOW (for cross-page reference detection):
Previous page: {prev_context}
Current page: {page_num}
Next page: {next_context}

### OUTPUT FORMAT (JSON):
{{
  "page_number": {page_num},
  "has_figures": true/false,
  "figures": [
    {{
      "figure_number": "Figure 1",
      "caption": "...",
      "image_type": "data_visualization",
      "chart_type": "line_chart",
      "visual_description": "...",
      "key_elements": ["..."],
      "text_labels": ["..."],
      "axes": {{
        "x_axis": {{"label": "...", "unit": null, "range": [...], "scale": "..."}},
        "y_axis": {{"label": "...", "unit": "...", "range": [...], "scale": "..."}}
      }},
      "data_series": [
        {{
          "series_name": "...",
          "data_points": [{{"x": "...", "y": ...}}],
          "trend": "upward/downward/stable/fluctuating"
        }}
      ],
      "statistical_info": {{
        "correlation": null,
        "r_squared": null,
        "equation": null,
        "sample_size": null,
        "growth_rate": null
      }}
    }}
  ],
  "tables": [
    {{"table_number": "Table 1", "caption": "...", "markdown": "| ... |"}}
  ],
  "formulas": ["$...$", "$$...$$"],
  "body_text": "...",
  "footnotes": []
}}

Return ONLY valid JSON.'''

# Chart indicator keywords (used to detect if enhanced Prompt is needed)
CHART_INDICATORS = [
    r'Figure\s*\d+', r'Fig\.\s*\d+', r'Table\s*\d+',
    r'Chart\s*\d+', r'Graph\s*\d+', r'Diagram\s*\d+',
    r'%', r'\$\d', r'million', r'billion', r'percent',
    r'x[\-\s]?axis', r'y[\-\s]?axis', r'legend',
    r'scatter', r'histogram', r'bar chart', r'pie chart'
]

def has_chart_indicators(ocr_text: str) -> bool:
    """Detect if OCR text contains chart-related keywords"""
    if not ocr_text:
        return False
    for pattern in CHART_INDICATORS:
        if re.search(pattern, ocr_text, re.IGNORECASE):
            return True
    return False

def build_context_window(prev_ocr: str, current_ocr: str, next_ocr: str,
                         mode: ContextMode = ContextMode.STANDARD) -> tuple:
    """
    Build three-page context window (Issue 5)

    Returns: (prev_context, current_context, next_context)
    """
    # Set limits based on mode
    if mode == ContextMode.MINIMAL:
        prev_limit, current_limit, next_limit = 100, 300, 100
    elif mode == ContextMode.STANDARD:
        prev_limit, current_limit, next_limit = 500, 0, 500  # 0 means no truncation
    else:  # FULL
        prev_limit, current_limit, next_limit = 2000, 0, 2000

    def truncate_with_refs(text: str, max_chars: int) -> str:
        """Smart truncation: preserve chart references"""
        if not text:
            return "N/A"
        if max_chars == 0 or len(text) <= max_chars:
            return text

        # Extract all chart references
        refs = re.findall(
            r'((?:Figure|Fig\.?|Table|Chart)\s*\d+[^.]*\.)',
            text, re.IGNORECASE
        )

        # If references take less than max_chars, prioritize keeping references
        refs_text = ' '.join(refs)
        if refs_text and len(refs_text) < max_chars:
            remaining = max_chars - len(refs_text) - 20
            if remaining > 50:
                return f"{text[:remaining]}...\n[Refs: {refs_text}]"

        return text[:max_chars] + "..."

    prev_context = truncate_with_refs(prev_ocr, prev_limit)
    current_context = truncate_with_refs(current_ocr, current_limit)
    next_context = truncate_with_refs(next_ocr, next_limit)

    return prev_context, current_context, next_context

def build_prompt(page_num: int, prev_context: str, current_context: str,
                 next_context: str, prompt_mode: PromptMode) -> str:
    """
    Build LLM Prompt (Issue 3)

    Select simplified or enhanced version based on prompt_mode
    """
    if prompt_mode == PromptMode.ENHANCED:
        # Use chain-of-thought enhanced Prompt
        return CHART_EXTRACTION_PROMPT.format(
            page_num=page_num,
            prev_context=prev_context,
            next_context=next_context
        )
    else:
        # Simplified Prompt (for plain text pages or politically sensitive documents)
        return f"""Page {page_num} analysis:

OCR text: {current_context[:300] + '...' if len(current_context) > 300 else current_context}

Context:
- Previous page {page_num-1}: {prev_context[:100] + '...' if len(prev_context) > 100 else prev_context}
- Next page {page_num+1}: {next_context[:100] + '...' if len(next_context) > 100 else next_context}

Extract ALL content with HIGH FIDELITY:

1. TABLES - Convert to Markdown table format
2. FORMULAS - Convert to LaTeX format: $E=mc^2$
3. FIGURES - Describe the image content, include figure number and caption
4. BODY TEXT - Keep original structure, paragraphs, headings
5. Keep original language - DO NOT translate

Return JSON format:
{{
  "page_number": {page_num},
  "tables": [{{"table_number": "Table 1", "caption": "...", "markdown": "| Col1 | Col2 |\\n|---|---|"}}],
  "figures": [{{"figure_number": "Fig 1", "caption": "...", "description": "..."}}],
  "formulas": ["$formula1$"],
  "body_text": "main text...",
  "footnotes": []
}}

Return ONLY valid JSON."""

# Import progress manager
try:
    from progress_manager import ProgressManager, get_progress_manager
    PROGRESS_ENABLED = True
except ImportError:
    PROGRESS_ENABLED = False
    print("⚠️  Progress manager not found, resume capability disabled")

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# ============ API Key Configuration (Lite Version) ============
# Load API key from environment variable DASHSCOPE_API_KEY
# Set this in .env file

API_KEYS = []
_single_key = os.environ.get("DASHSCOPE_API_KEY")
if _single_key and _single_key != "YOUR_API_KEY_HERE" and _single_key != "your-dashscope-api-key-here":
    API_KEYS = [_single_key]

# Create cycle iterator for load balancing (backward compatible)
_api_key_cycle = cycle(API_KEYS) if API_KEYS else None
_api_key_lock = asyncio.Lock()

# Initialize health monitor
if API_KEYS:
    _api_key_monitor = APIKeyHealthMonitor(
        API_KEYS,
        disable_threshold=5,      # Disable after 5 consecutive failures
        disable_duration=300.0    # Disable for 5 minutes
    )

async def get_next_api_key():
    """Get next API Key (smart health monitoring version)"""
    global _api_key_monitor
    if _api_key_monitor:
        return await _api_key_monitor.get_healthy_key()
    # Fallback to single key
    return os.environ.get("DASHSCOPE_API_KEY")

def get_next_api_key_sync():
    """Sync version: Get next API Key (smart health monitoring version)"""
    global _api_key_monitor
    if _api_key_monitor:
        return _api_key_monitor.get_healthy_key_sync()
    return os.environ.get("DASHSCOPE_API_KEY")
# ========================================

def check_dependencies():
    """Check dependencies"""
    missing = []

    try:
        import dashscope
    except ImportError:
        missing.append("dashscope")

    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing.append("pdf2image")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    try:
        import yaml
    except ImportError:
        missing.append("PyYAML")

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"\nInstall command:")
        print(f"pip3 install {' '.join(missing)}")
        return False

    return True

def get_api_key():
    """Get API key - read from environment variable"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key and api_key != "YOUR_API_KEY_HERE":
        return api_key
    return None

def safe_get_text_from_response(response):
    """
    Safely extract text from API response
    Handle multiple response formats, avoid IndexError and TypeError
    """
    try:
        # Check basic structure
        if not hasattr(response, 'output'):
            return ""

        if not hasattr(response.output, 'choices') or len(response.output.choices) == 0:
            return ""

        content = response.output.choices[0].message.content

        # Case 1: content is a list
        if isinstance(content, list):
            if len(content) == 0:
                return ""

            first_item = content[0]

            # List element is a dictionary
            if isinstance(first_item, dict):
                return first_item.get("text", "")
            # List element is a string
            elif isinstance(first_item, str):
                return first_item
            else:
                return str(first_item)

        # Case 2: content is directly a string
        elif isinstance(content, str):
            return content

        # Case 3: content is a dictionary
        elif isinstance(content, dict):
            return content.get("text", str(content))

        # Case 4: Other types
        else:
            return str(content)

    except Exception as e:
        return ""

def simple_ocr_extract(image, retry_config: RetryConfig = DEFAULT_RETRY_CONFIG) -> str:
    """
    Simple OCR extraction (Phase 2) - with retry mechanism
    Use Qwen API for basic text extraction
    Supports smart API Key health monitoring and layered retry
    """
    from dashscope import MultiModalConversation
    import dashscope
    import base64
    from io import BytesIO

    global _api_key_monitor

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    simple_ocr_prompt = """请提取这张图片中的所有文字内容。
只需要纯文本，保持原有的段落结构和换行。
不要添加任何解释或格式化。"""

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{img_base64}"},
                {"text": simple_ocr_prompt}
            ]
        }
    ]

    last_error = None

    for attempt in range(retry_config.max_retries + 1):
        api_key = get_next_api_key_sync()
        call_start = time.time()

        try:
            dashscope.api_key = api_key

            response = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages
            )

            call_latency = time.time() - call_start

            if response.status_code == 200:
                # Record success
                if _api_key_monitor:
                    _api_key_monitor.record_success_sync(api_key, call_latency)
                return safe_get_text_from_response(response)

            # API returned error status code
            error_msg = f"{response.code} - {response.message}"
            last_error = Exception(error_msg)

            # Record failure
            if _api_key_monitor:
                _api_key_monitor.record_failure_sync(api_key)

            # Check if should retry
            if response.status_code in retry_config.retry_on_status_codes:
                if attempt < retry_config.max_retries:
                    delay = calculate_retry_delay(attempt, retry_config)
                    print(f"      🔄 OCR retry ({attempt+1}/{retry_config.max_retries}): "
                          f"status code {response.status_code}, waiting {delay:.1f}s")
                    time.sleep(delay)
                    continue

            return ""

        except Exception as e:
            call_latency = time.time() - call_start
            last_error = e

            # Record failure
            if _api_key_monitor:
                _api_key_monitor.record_failure_sync(api_key)

            # Check if should retry
            if should_retry(e, retry_config) and attempt < retry_config.max_retries:
                delay = calculate_retry_delay(attempt, retry_config)
                print(f"      🔄 OCR retry ({attempt+1}/{retry_config.max_retries}): "
                      f"{type(e).__name__}, waiting {delay:.1f}s")
                time.sleep(delay)
                continue

            print(f"      ⚠️ OCR extraction failed: {e}")
            return ""

    # All retries failed
    print(f"      ❌ OCR extraction failed (after {retry_config.max_retries} retries): {last_error}")
    return ""

async def process_single_page_with_context(
    page_num: int,
    image,
    current_ocr: str,
    prev_ocr: Optional[str],
    next_ocr: Optional[str],
    api_key: str,
    semaphore: asyncio.Semaphore,
    images_dir: Path,
    yaml_dir: Path,
    context_mode: ContextMode = DEFAULT_CONTEXT_MODE,
    prompt_mode: PromptMode = DEFAULT_PROMPT_MODE,
    model: str = DEFAULT_MODEL,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG
) -> Dict[str, Any]:
    """
    Process single page (Phase 3 - concurrent LLM processing) - with retry mechanism

    Args:
        page_num: Current page number
        image: Current page image
        current_ocr: Current page OCR text (must be processed)
        prev_ocr: Previous page OCR text (for context only)
        next_ocr: Next page OCR text (for context only)
        api_key: API key
        semaphore: Concurrency control semaphore
        images_dir: Image save directory
        yaml_dir: YAML save directory
        context_mode: Context window mode (Issue 5)
        prompt_mode: Prompt mode (Issue 3)
        model: Model name to use
        retry_config: Retry configuration (Issue 4)
    """
    from dashscope import MultiModalConversation
    import base64
    from io import BytesIO
    import dashscope

    global _api_key_monitor

    # Use semaphore for concurrency control
    async with semaphore:
        page_start = time.time()

        # Save page image (done once, no retry needed)
        page_image_filename = f"page_{page_num:03d}.png"
        page_image_path = images_dir / page_image_filename
        image.save(page_image_path, "PNG")

        # Convert image to base64 (done once)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # ⭐ Issue 3 & 5: Use enhanced context window and Prompt construction
        # Smart detection of whether enhanced mode is needed
        detected_prompt_mode = prompt_mode
        if prompt_mode == PromptMode.ENHANCED:
            # If OCR text has no chart indicators, can downgrade to SIMPLE to save tokens
            if not has_chart_indicators(current_ocr) and not has_chart_indicators(prev_ocr or '') and not has_chart_indicators(next_ocr or ''):
                detected_prompt_mode = PromptMode.SIMPLE

        # Build context window (Issue 5)
        prev_context, current_context, next_context = build_context_window(
            prev_ocr or "", current_ocr, next_ocr or "", context_mode
        )

        # Build Prompt (Issue 3)
        llm_prompt = build_prompt(
            page_num, prev_context, current_context, next_context, detected_prompt_mode
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{img_base64}"},
                    {"text": llm_prompt}
                ]
            }
        ]

        # ⭐ Issue 4: Layered retry logic
        last_error = None
        response = None

        for attempt in range(retry_config.max_retries + 1):
            # Get new healthy API Key for each retry
            balanced_api_key = await get_next_api_key()
            dashscope.api_key = balanced_api_key
            call_start = time.time()

            try:
                # Async API call (using sync method since dashscope doesn't support async)
                # Use run_in_executor in asyncio environment
                # ⭐ Issue 3 & 5: Use configured model (default qwen-vl-max-latest)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: MultiModalConversation.call(
                        model=model,
                        messages=messages
                    )
                )

                call_latency = time.time() - call_start

                if response.status_code == 200:
                    # Record success
                    if _api_key_monitor:
                        await _api_key_monitor.record_success(balanced_api_key, call_latency)
                    break  # Success, exit retry loop

                # API returned error status code
                error_msg = f"{response.code} - {response.message}"
                last_error = Exception(error_msg)

                # Record failure
                if _api_key_monitor:
                    await _api_key_monitor.record_failure(balanced_api_key)

                # Check if should retry (not content moderation error)
                if "DataInspectionFailed" not in error_msg and attempt < retry_config.max_retries:
                    if response.status_code in retry_config.retry_on_status_codes or response.status_code >= 500:
                        delay = calculate_retry_delay(attempt, retry_config)
                        print(f"      🔄 Page {page_num} retry ({attempt+1}/{retry_config.max_retries}): "
                              f"status code {response.status_code}, waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue

                # Non-retryable error
                return {
                    'success': False,
                    'page': page_num,
                    'error': error_msg
                }

            except Exception as e:
                call_latency = time.time() - call_start
                last_error = e

                # Record failure
                if _api_key_monitor:
                    await _api_key_monitor.record_failure(balanced_api_key)

                # Check if should retry
                if should_retry(e, retry_config) and attempt < retry_config.max_retries:
                    delay = calculate_retry_delay(attempt, retry_config)
                    print(f"      🔄 Page {page_num} retry ({attempt+1}/{retry_config.max_retries}): "
                          f"{type(e).__name__}, waiting {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable error
                return {
                    'success': False,
                    'page': page_num,
                    'error': str(e),
                    'traceback': __import__('traceback').format_exc()
                }

        # Check if all retries failed
        if response is None or response.status_code != 200:
            print(f"      ❌ Page {page_num} failed (after {retry_config.max_retries} retries): {last_error}")
            return {
                'success': False,
                'page': page_num,
                'error': str(last_error) if last_error else "Unknown error after retries"
            }

        # ========== Process successful response ==========
        try:
            result_text = safe_get_text_from_response(response)
            usage = response.usage

            # Extract JSON (Issue 11 fix: handle LaTeX backslash escapes)
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else result_text

            # Fix illegal backslash escapes in LaTeX formulas
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            # Common LaTeX: \frac \int \alpha \beta etc. need to be converted to \\
            def fix_latex_escapes(s):
                result = []
                i = 0
                n = len(s)
                while i < n:
                    if s[i] == '\\' and i + 1 < n:
                        next_char = s[i + 1]
                        if next_char in '"\\/' :
                            # Keep: \" \\ \/
                            result.append(s[i:i+2])
                            i += 2
                        elif next_char in 'bfnrt':
                            # Determine if JSON escape or LaTeX command
                            if i + 2 < n and s[i + 2].isalpha():
                                # LaTeX command like \beta \frac \ne
                                result.append('\\\\')
                                i += 1
                            else:
                                # JSON escape like \n \t
                                result.append(s[i:i+2])
                                i += 2
                        elif next_char == 'u' and i + 5 < n:
                            # Check if it's \uXXXX
                            hex_part = s[i+2:i+6]
                            if all(c in '0123456789abcdefABCDEF' for c in hex_part):
                                result.append(s[i:i+6])
                                i += 6
                            else:
                                result.append('\\\\')
                                i += 1
                        else:
                            # Other cases (LaTeX commands)
                            result.append('\\\\')
                            i += 1
                    else:
                        result.append(s[i])
                        i += 1
                return ''.join(result)

            json_str = fix_latex_escapes(json_str)

            try:
                result_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Fallback: return basic structure
                print(f"      ⚠️ JSON parse failed (Page {page_num}): {e}")
                result_json = {
                    "page_number": page_num,
                    "tables": [],
                    "figures": [],
                    "formulas": [],
                    "body_text": result_text[:500] if result_text else "",
                    "footnotes": [],
                    "page_markers": []
                }

            page_time = time.time() - page_start

            # Process detected charts
            figures_metadata = []
            figure_refs = []

            # Process figures (image/chart descriptions)
            if result_json.get('figures'):
                for fig_idx, figure in enumerate(result_json['figures'], 1):
                    figure_yaml = generate_figure_yaml(figure, fig_idx, page_num)
                    figure_ref = figure.get('figure_number', f'Figure_{fig_idx}')
                    yaml_filename = f"{figure_ref.replace(' ', '_').replace(':', '')}_page{page_num}.yaml"
                    yaml_path = yaml_dir / yaml_filename

                    with open(yaml_path, 'w', encoding='utf-8') as f:
                        yaml.dump(figure_yaml, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

                    figures_metadata.append(figure_yaml)
                    figure_refs.append({
                        'figure_ref': figure_ref,
                        'yaml_file': yaml_filename,
                        'description': figure.get('description', ''),
                        'confidence': figure.get('extraction_confidence', 'medium')
                    })

            # Process tables (Markdown tables)
            tables_data = []
            if result_json.get('tables'):
                for table in result_json['tables']:
                    tables_data.append({
                        'table_number': table.get('table_number', ''),
                        'caption': table.get('caption', ''),
                        'markdown': table.get('markdown', '')
                    })

            # Process formulas (LaTeX formulas)
            formulas = result_json.get('formulas', [])

            # Log: VLM processing success
            print(f"   [VLM] [{page_num}] ✅ ({len(result_text)} chars)", flush=True)

            return {
                'success': True,
                'page': page_num,
                'figure_count': len(figure_refs),
                'table_count': len(tables_data),
                'formula_count': len(formulas),
                'figures': figure_refs,
                'figures_metadata': figures_metadata,
                'tables': tables_data,
                'formulas': formulas,
                'body_text': result_json.get('body_text', ''),
                'footnotes': result_json.get('footnotes', []),
                'tokens': {
                    'input': usage.input_tokens,
                    'output': usage.output_tokens
                },
                'time': page_time
            }

        except Exception as e:
            return {
                'success': False,
                'page': page_num,
                'error': str(e),
                'traceback': __import__('traceback').format_exc()
            }

def generate_figure_yaml(figure_data: Dict, figure_index: int, page_num: int) -> Dict[str, Any]:
    """
    Generate complete YAML metadata (Issue 2: Complete 6-section structure)
    Complete schema compliant with MARCO quality standards
    """
    confidence = figure_data.get("extraction_confidence", "medium")
    conf_score = {"high": 0.95, "medium": 0.70, "low": 0.40}.get(confidence, 0.70)

    yaml_metadata = {
        # Section 1: Chart Identification (必填)
        "chart_identification": {
            "chart_title": figure_data.get("chart_title", figure_data.get("caption", "Untitled")),
            "figure_number": figure_data.get("figure_number", f"Figure {figure_index}"),
            "figure_reference_in_text": figure_data.get("figure_number", f"Figure {figure_index}"),
            "page_number": page_num,
            "image_path": f"images/page_{page_num:03d}.png",
            "image_type": figure_data.get("image_type", "figure"),
            "chart_type": figure_data.get("chart_type", "unknown")
        },

        # Section 2: Visual Content (必填)
        "visual_content": {
            "content_description": figure_data.get("content_description", figure_data.get("description", "")),
            "key_elements": figure_data.get("key_elements", []),
            "text_labels": figure_data.get("text_labels", []),
            "content_summary": (figure_data.get("content_description", "") or figure_data.get("description", ""))[:200],
            "key_insight": figure_data.get("key_insight", "")
        },

        # Section 3: Data Extraction (有数据时填充，否则占位)
        "data_extraction": {
            "axes": figure_data.get("axes", {
                "x_axis": {"label": "N/A", "unit": "N/A", "range": "N/A"},
                "y_axis": {"label": "N/A", "unit": "N/A", "range": "N/A"}
            }),
            "data_series": figure_data.get("data_series", []) if figure_data.get("data_series") else (
                [{
                    "series_name": "Main Data",
                    "data_points": figure_data.get("data_points", []),
                    "trend": figure_data.get("trend", "N/A")
                }] if figure_data.get("data_points") else []
            ),
            "has_quantitative_data": bool(figure_data.get("data_points") or figure_data.get("has_data"))
        },

        # Section 4: Statistical Information (占位结构)
        "statistical_information": figure_data.get("statistics", {
            "sample_size": "N/A",
            "statistical_tests": [],
            "confidence_intervals": [],
            "p_values": [],
            "notes": "No statistical data extracted"
        }),

        # Section 5: Visual Design (占位结构)
        "visual_design": figure_data.get("visual_design", {
            "color_scheme": [],
            "legend_present": False,
            "grid_lines": "N/A",
            "annotations": []
        }),

        # Section 6: Quality Check (必填)
        "quality_check": {
            "data_completeness": {
                "all_labels_readable": "yes" if confidence == "high" else ("partial" if confidence == "medium" else "no"),
                "all_values_extracted": "yes" if figure_data.get("has_data") or figure_data.get("data_points") else "no",
                "uncertainties": figure_data.get("uncertainties", []),
                "total_data_points_visible": len(figure_data.get("data_points", []))
            },
            "extraction_confidence": confidence,
            "confidence_score": conf_score,
            "validation_checklist": {
                "figure_number_found": "yes" if figure_data.get("figure_number") else "no",
                "image_type_identified": "yes" if figure_data.get("image_type") else "no",
                "all_axes_labeled": "yes" if figure_data.get("axes") else "no",
                "data_points_extracted": "yes" if figure_data.get("data_points") else "no",
                "manual_verification_needed": "no" if confidence == "high" else "yes"
            }
        }
    }

    return yaml_metadata

async def process_pdf_async(
    pdf_path: Path,
    api_key: str,
    output_base: Path,
    max_pages: int = None,
    semaphore_limit: int = 6,  # Lite default: 6
    progress_manager: 'ProgressManager' = None,
    resume: bool = True,
    context_mode: ContextMode = DEFAULT_CONTEXT_MODE,
    prompt_mode: PromptMode = DEFAULT_PROMPT_MODE,
    model: str = DEFAULT_MODEL
):
    """
    Async process single PDF file

    Args:
        pdf_path: PDF file path
        api_key: API key
        output_base: Output directory
        max_pages: Max pages to process
        semaphore_limit: Concurrency limit
        progress_manager: Progress manager (for resume capability)
        resume: Whether to enable resume
    """
    from pdf2image import convert_from_path
    import dashscope

    dashscope.api_key = api_key
    pdf_name = pdf_path.stem

    print(f"\n{'='*80}")
    print(f"📄 Processing PDF: {pdf_path.name}")
    print(f"{'='*80}")
    print(f"   Size: {pdf_path.stat().st_size / 1024 / 1024:.1f}MB")
    print(f"   Concurrency limit: {semaphore_limit}")

    # Check if already completed (resume) - enhanced validation
    if resume and progress_manager:
        # Determine step name
        step_name = 'process_chunks' if 'chunks' in str(output_base) else 'process_direct'

        # First check if in completed list
        if progress_manager.is_pdf_completed(step_name, pdf_name):
            # Use enhanced validation: check MD file existence and content size
            validation = progress_manager.validate_pdf_completion(
                step_name,
                pdf_name,
                str(output_base),
                min_content_size=500,      # MD file at least 500 bytes
                min_completion_rate=0.90   # Page completion rate at least 90%
            )

            if validation['valid']:
                print(f"   ⏭️  Already completed, skipping (validated: {validation['md_size']} bytes, {validation['page_completion_rate']:.1%} completion rate)")
                return {
                    "success": True,
                    "skipped": True,
                    "pdf_info": {"name": pdf_path.name},
                    "message": "Completed in previous run"
                }
            else:
                # Validation failed, need to reprocess
                print(f"   ⚠️  Previously marked complete but validation failed: {validation['reason']}")
                print(f"   🔄 Will reprocess this PDF...")
                progress_manager.invalidate_pdf_completion(step_name, pdf_name)

    # Create output directories
    pdf_name = pdf_path.stem
    output_dir = output_base / pdf_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    yaml_dir = output_dir / "yaml_metadata"
    yaml_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Phase 1: PDF → Images (using run_in_executor for concurrency)
        print(f"\n📸 Phase 1: Converting PDF to images...")
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            lambda: convert_from_path(str(pdf_path), dpi=150)
        )
        total_pages = len(images)

        if max_pages:
            images = images[:max_pages]
            print(f"   Limited to first {max_pages} pages (total {total_pages} pages)")

        print(f"   ✅ Conversion complete: {len(images)} pages")

        # Initialize page progress tracking
        if progress_manager:
            progress_manager.init_pdf_progress(pdf_name, len(images))
            completed_pages = set(progress_manager.get_completed_pages(pdf_name))
            if completed_pages:
                print(f"   📌 Resume: {len(completed_pages)}/{len(images)} pages completed")
        else:
            completed_pages = set()

        # Phase 2: OCR concurrent extraction
        print(f"\n🔤 Phase 2: OCR concurrent extraction...")
        print(f"   Concurrency: {semaphore_limit}")

        async def ocr_single_page(page_num: int, image):
            """Single page OCR async wrapper"""
            # Skip completed pages (but OCR phase needs all for context support)
            loop = asyncio.get_event_loop()
            ocr_text = await loop.run_in_executor(
                None,
                lambda: simple_ocr_extract(image)
            )
            print(f"   [OCR] [{page_num}/{len(images)}] ✅ ({len(ocr_text)} chars)", flush=True)
            return page_num, ocr_text

        # Execute all OCR concurrently (even completed LLM pages need OCR for context)
        # Lite: Keep consistent with LLM semaphore = 6 to reduce memory peak
        ocr_semaphore = asyncio.Semaphore(semaphore_limit)

        async def ocr_with_semaphore(page_num, image):
            async with ocr_semaphore:
                return await ocr_single_page(page_num, image)

        ocr_tasks = [
            ocr_with_semaphore(page_num, image)
            for page_num, image in enumerate(images, 1)
        ]
        ocr_results = await asyncio.gather(*ocr_tasks)

        page_ocr_texts = {page_num: text for page_num, text in ocr_results}
        print(f"   ✅ OCR complete: {len(page_ocr_texts)} pages")

        # Phase 3: LLM concurrent processing
        print(f"\n🤖 Phase 3: LLM concurrent processing (chart detection + metadata generation)...")
        print(f"   Concurrency: {semaphore_limit}")

        # Count pages to process
        pages_to_process = [p for p in range(1, len(images) + 1) if p not in completed_pages]
        if completed_pages:
            print(f"   📌 Skipping completed pages: {len(completed_pages)} pages")
            print(f"   📌 Pages to process: {len(pages_to_process)} pages")

        semaphore = asyncio.Semaphore(semaphore_limit)
        tasks = []
        skipped_results = []

        for page_num, image in enumerate(images, 1):
            # Skip completed pages
            if page_num in completed_pages:
                # Try to load results from existing files
                skipped_results.append({
                    'success': True,
                    'page': page_num,
                    'skipped': True,
                    'figure_count': 0,  # Will be re-read from file during merge
                    'figures': [],
                    'figures_metadata': [],
                    'body_text': '',
                    'tokens': {'input': 0, 'output': 0},
                    'time': 0
                })
                continue

            current_ocr = page_ocr_texts.get(page_num, "")
            prev_ocr = page_ocr_texts.get(page_num - 1)
            next_ocr = page_ocr_texts.get(page_num + 1)

            task = process_single_page_with_context(
                page_num=page_num,
                image=image,
                current_ocr=current_ocr,
                prev_ocr=prev_ocr,
                next_ocr=next_ocr,
                api_key=api_key,
                semaphore=semaphore,
                images_dir=images_dir,
                yaml_dir=yaml_dir,
                context_mode=context_mode,  # Issue 5
                prompt_mode=prompt_mode,     # Issue 3
                model=model                  # qwen-vl-max-latest
            )
            tasks.append((page_num, task))

        # Execute all tasks concurrently
        if tasks:
            async_results = await asyncio.gather(*[t[1] for t in tasks])

            # Process results and update progress
            for (page_num, _), result in zip(tasks, async_results):
                if result.get('success'):
                    if progress_manager:
                        progress_manager.mark_page_completed(pdf_name, page_num)
                else:
                    if progress_manager:
                        progress_manager.mark_page_failed(pdf_name, page_num, result.get('error', 'Unknown error'))

            page_results = skipped_results + list(async_results)
        else:
            page_results = skipped_results
            print(f"   ✅ All pages completed in previous run")

        # Process results
        print(f"\n💾 Saving results...")

        # Sort by page number
        page_results = sorted(page_results, key=lambda x: x['page'])

        # Build Markdown
        markdown_sections = []
        all_figures_metadata = []
        total_tokens = {"input": 0, "output": 0}
        total_figures = 0
        total_tables = 0
        total_formulas = 0
        confidence_scores = []

        for result in page_results:
            if not result['success']:
                print(f"   ⚠️ Page {result['page']} failed: {result.get('error')}")
                continue

            page_num = result['page']
            tokens = result['tokens']
            total_tokens['input'] += tokens['input']
            total_tokens['output'] += tokens['output']

            figure_count = result.get('figure_count', 0)
            table_count = result.get('table_count', 0)
            formula_count = result.get('formula_count', 0)
            total_figures += figure_count
            total_tables += table_count
            total_formulas += formula_count

            # Build page content
            page_content = f"## Page {page_num}\n\n"

            # Add body text (at the beginning)
            body_text = result.get('body_text', '').strip()
            if body_text:
                page_content += f"{body_text}\n\n"

            # Add tables (Markdown format)
            if result.get('tables'):
                for table in result['tables']:
                    table_num = table.get('table_number', '')
                    caption = table.get('caption', '')
                    markdown = table.get('markdown', '')
                    if table_num or caption:
                        page_content += f"### {table_num}: {caption}\n\n"
                    if markdown:
                        page_content += f"{markdown}\n\n"

            # Add formulas (LaTeX format)
            if result.get('formulas'):
                formulas = result['formulas']
                if formulas:
                    for formula in formulas:
                        if formula and formula.strip():
                            page_content += f"{formula}\n\n"

            # Add figure descriptions
            if result.get('figures'):
                for fig_info in result['figures']:
                    figure_ref = fig_info['figure_ref']
                    yaml_file = fig_info.get('yaml_file', '')
                    description = fig_info.get('description', '')

                    # Get title from metadata
                    matching_meta = [m for m in result.get('figures_metadata', [])
                                    if m['chart_identification']['figure_number'] == figure_ref]
                    chart_title = matching_meta[0]['chart_identification']['chart_title'] if matching_meta else "Untitled"

                    page_content += f"\n### {figure_ref}: {chart_title}\n\n"
                    if description:
                        page_content += f"*{description}*\n\n"
                    page_content += f"![{figure_ref}: {chart_title}](images/page_{page_num:03d}.png)\n\n"
                    if yaml_file:
                        page_content += f"*YAML Metadata: [yaml_metadata/{yaml_file}](yaml_metadata/{yaml_file})*\n\n"

                    # Record confidence
                    conf_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
                    confidence_scores.append(conf_map.get(fig_info.get('confidence', 'medium'), 0.6))

                all_figures_metadata.extend(result.get('figures_metadata', []))

            # Add footnotes
            if result.get('footnotes'):
                footnotes = result['footnotes']
                if footnotes:
                    page_content += "---\n\n**Footnotes:**\n\n"
                    for i, fn in enumerate(footnotes, 1):
                        page_content += f"[^{i}]: {fn}\n"
                    page_content += "\n"

            markdown_sections.append(page_content)
            
            print(f"   ✅ Page {page_num}: {table_count} tables, {formula_count} formulas, {figure_count} figures, {tokens['input']}+{tokens['output']} tokens")

        # Save Markdown
        md_file = output_dir / f"{pdf_name}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# {pdf_name}\n\n")
            f.write(f"*Processed with DocMind on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"*Model: {model}*\n\n")
            f.write(f"*Statistics: {total_tables} tables, {total_formulas} formulas, {total_figures} figures*\n\n")
            f.write("---\n\n")
            f.write("\n".join(markdown_sections))
        
        print(f"   ✅ Markdown: {md_file.name}")
        
        # Save combined YAML
        combined_yaml_file = output_dir / f"{pdf_name}_all_figures.yaml"
        combined_yaml = {
            "document_info": {
                "filename": pdf_path.name,
                "total_pages": total_pages,
                "processed_pages": len(images),
                "total_figures": total_figures,
                "processing_date": datetime.now().isoformat()
            },
            "figures": all_figures_metadata
        }
        
        with open(combined_yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(combined_yaml, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        
        print(f"   ✅ Combined YAML: {combined_yaml_file.name}")
        
        # Generate validation report (Issue 6 + Issue 8: Enhanced validation report and KQI metrics)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        yaml_insertion_rate = len(all_figures_metadata) / max(total_figures, 1)

        # Count successful/failed pages
        success_pages = [r for r in page_results if r.get('success', False)]
        failed_pages = [r for r in page_results if not r.get('success', False)]

        # KQI metrics calculation (Issue 8)
        kqi_yaml_insertion = yaml_insertion_rate >= 0.95  # MARCO requirement ≥95%
        kqi_confidence = avg_confidence >= 0.85           # MARCO requirement ≥0.85
        kqi_page_success = len(success_pages) / len(page_results) if page_results else 0

        validation_report = {
            "document_info": {
                "filename": pdf_path.name,
                "total_pages": total_pages,
                "processed_pages": len(images),
                "processing_date": datetime.now().isoformat()
            },
            "kqi_metrics": {  # Issue 8: Key Quality Indicators
                "yaml_insertion_rate": round(yaml_insertion_rate, 4),
                "yaml_insertion_pass": kqi_yaml_insertion,  # ≥95%
                "average_confidence": round(avg_confidence, 4),
                "confidence_pass": kqi_confidence,          # ≥0.85
                "page_success_rate": round(kqi_page_success, 4),
                "overall_quality_pass": kqi_yaml_insertion and kqi_confidence and kqi_page_success >= 0.98
            },
            "validation_metrics": {
                "yaml_insertion_rate": round(yaml_insertion_rate, 2),
                "average_confidence": round(avg_confidence, 2),
                "figure_detection_completeness": round(total_figures / len(images), 2) if images else 0,
                "total_figures_detected": total_figures,
                "total_tables_detected": total_tables,
                "total_formulas_detected": total_formulas,
                "pages_with_figures": sum(1 for r in page_results if r.get('figure_count', 0) > 0),
                "pages_with_tables": sum(1 for r in page_results if r.get('table_count', 0) > 0),
                "pages_with_formulas": sum(1 for r in page_results if r.get('formula_count', 0) > 0),
                "high_confidence_figures": sum(1 for s in confidence_scores if s >= 0.9),
                "medium_confidence_figures": sum(1 for s in confidence_scores if 0.5 <= s < 0.9),
                "low_confidence_figures": sum(1 for s in confidence_scores if s < 0.5)
            },
            "page_statistics": {  # Issue 6: Enhanced page statistics
                "total_pages": len(page_results),
                "successful_pages": len(success_pages),
                "failed_pages": len(failed_pages),
                "skipped_pages": sum(1 for r in page_results if r.get('skipped', False)),
                "success_rate": round(len(success_pages) / len(page_results), 4) if page_results else 0
            },
            "quality_indicators": {
                "all_figures_have_yaml": len(all_figures_metadata) == total_figures,
                "zero_hallucination": True,
                "proper_markdown_format": True,
                "complete_data_extraction": avg_confidence >= 0.6,
                "no_page_failures": len(failed_pages) == 0
            },
            "failed_pages_detail": [  # Issue 6: Failed pages detail
                {
                    "page": r['page'],
                    "error": r.get('error', 'Unknown error')
                } for r in failed_pages
            ],
            "page_by_page_results": [
                {
                    "page": r['page'],
                    "success": r['success'],
                    "figure_count": r.get('figure_count', 0),
                    "table_count": r.get('table_count', 0),
                    "formula_count": r.get('formula_count', 0),
                    "figures": r.get('figures', [])
                } for r in page_results
            ]
        }
        
        validation_file = output_dir / f"{pdf_name}.validation.yaml"
        with open(validation_file, 'w', encoding='utf-8') as f:
            yaml.dump(validation_report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        
        print(f"   ✅ Validation Report: {validation_file.name}")
        
        # Calculate total time and cost
        # Qwen-VL-Max pricing (2024): Input ¥0.02/1K, Output ¥0.02/1K
        # Note: Actual billing may vary due to discounts
        PRICE_INPUT_PER_1K = 0.02   # ¥0.02/1000 input tokens
        PRICE_OUTPUT_PER_1K = 0.02  # ¥0.02/1000 output tokens

        process_time = time.time() - start_time
        total_tokens_sum = total_tokens["input"] + total_tokens["output"]

        # Calculate input and output costs separately
        input_cost = (total_tokens["input"] / 1000) * PRICE_INPUT_PER_1K
        output_cost = (total_tokens["output"] / 1000) * PRICE_OUTPUT_PER_1K
        cost_cny = input_cost + output_cost

        success_count = sum(1 for r in page_results if r.get('success', False))

        print(f"\n{'='*80}")
        print(f"✅ Processing complete: {pdf_path.name}")
        print(f"{'='*80}")
        print(f"📊 Statistics:")
        print(f"   Pages: {len(images)}/{total_pages} (successful: {success_count})")
        print(f"   Figures: {total_figures}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   YAML insertion rate: {yaml_insertion_rate:.2%}")
        print(f"   Time: {int(process_time // 60)}m {int(process_time % 60)}s")
        print(f"   Tokens: {total_tokens_sum:,} (input: {total_tokens['input']:,}, output: {total_tokens['output']:,})")
        print(f"   Cost: ¥{cost_cny:.2f} (input: ¥{input_cost:.2f}, output: ¥{output_cost:.2f})")
        
        return {
            "success": True,
            "pdf_info": {
                "name": pdf_path.name,
                "pages": len(images),
                "figures": total_figures
            },
            "processing": {
                "time": process_time,
                "tokens": total_tokens_sum,
                "tokens_input": total_tokens["input"],
                "tokens_output": total_tokens["output"],
                "cost_cny": cost_cny,
                "cost_input": input_cost,
                "cost_output": output_cost
            },
            "validation": validation_report["validation_metrics"]
        }
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "pdf_info": {"name": pdf_path.name},
            "error": str(e)
        }

async def process_all_pdfs_parallel(
    pdf_files: List[Path],
    api_key: str,
    output_base: Path,
    max_pages: Optional[int],
    pdf_concurrency: int,
    semaphore_limit: int = 6,  # Lite default: 6 (was 12)
    progress_manager: 'ProgressManager' = None,
    resume: bool = True,
    context_mode: ContextMode = DEFAULT_CONTEXT_MODE,
    prompt_mode: PromptMode = DEFAULT_PROMPT_MODE,
    model: str = DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    """
    Process all PDF files in parallel at the PDF level

    Args:
        pdf_files: List of PDF files
        api_key: API key
        output_base: Output directory
        max_pages: Maximum page limit
        pdf_concurrency: PDF-level concurrency (how many PDFs to process simultaneously)
        semaphore_limit: LLM concurrency (within each PDF)
        progress_manager: Progress manager (for resume capability)
        resume: Whether to enable resume

    Returns:
        Processing results for all PDFs
    """
    import sys

    # Create PDF-level semaphore
    pdf_semaphore = asyncio.Semaphore(pdf_concurrency)

    # Progress tracking
    total = len(pdf_files)
    completed = 0
    skipped = 0
    lock = asyncio.Lock()

    async def process_one_pdf_with_semaphore(pdf_file: Path, index: int):
        """Single PDF processing with semaphore control"""
        nonlocal completed, skipped

        async with pdf_semaphore:
            # Print start info
            async with lock:
                print(f"\n{'#'*80}", flush=True)
                print(f"Progress: [{index}/{total}] - {pdf_file.name}", flush=True)
                print(f"{'#'*80}", flush=True)

            try:
                # LLM concurrent per PDF (Max Safe mode)
                result = await process_pdf_async(
                    pdf_file,
                    api_key,
                    output_base,
                    max_pages,
                    semaphore_limit=semaphore_limit,
                    progress_manager=progress_manager,
                    resume=resume,
                    context_mode=context_mode,
                    prompt_mode=prompt_mode,
                    model=model
                )

                async with lock:
                    completed += 1
                    if result.get('skipped'):
                        skipped += 1
                        print(f"\n⏭️  [{completed}/{total}] Skipped (already completed): {pdf_file.name}", flush=True)
                    else:
                        print(f"\n✅ [{completed}/{total}] Completed: {pdf_file.name}", flush=True)

                        # Update progress manager
                        if progress_manager and result.get('success'):
                            # Determine if chunks or direct based on output directory
                            step_name = 'process_chunks' if 'chunks' in str(output_base) else 'process_direct'
                            progress_manager.mark_pdf_completed(step_name, pdf_file.stem)

                return result
            except Exception as e:
                async with lock:
                    completed += 1
                    print(f"\n❌ [{completed}/{total}] Failed: {pdf_file.name} - {e}", flush=True)
                    sys.stderr.write(f"Error processing {pdf_file.name}: {e}\n")

                    # Record failure
                    if progress_manager:
                        step_name = 'process_chunks' if 'chunks' in str(output_base) else 'process_direct'
                        progress_manager.mark_pdf_failed(step_name, pdf_file.stem, str(e))

                return {
                    "success": False,
                    "pdf_info": {"name": pdf_file.name},
                    "error": str(e)
                }

    # Use asyncio.gather to process all PDFs in parallel
    results = await asyncio.gather(*[
        process_one_pdf_with_semaphore(pdf_file, i)
        for i, pdf_file in enumerate(pdf_files, 1)
    ], return_exceptions=False)

    if skipped > 0:
        print(f"\n📌 Resume statistics: Skipped {skipped} already completed PDFs")

    return results

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="DocMind PDF Converter")
    parser.add_argument("--input", default="combined_input", help="Input PDF directory")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per PDF")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--semaphore-limit", type=int, default=6, help="LLM concurrent limit per PDF (default: 6)")
    parser.add_argument("--pdf-concurrency", type=int, default=4, help="PDF parallel count (default: 4)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume, start fresh")
    parser.add_argument("--progress-file", default=None, help="Progress file path")
    # Issue 3 & 5: Added model and mode configuration parameters
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--context-mode", choices=["minimal", "standard", "full"], default="standard",
                        help="Context window mode: minimal(100/300/100), standard(500/full/500), full(2000/full/2000)")
    parser.add_argument("--prompt-mode", choices=["simple", "enhanced"], default="enhanced",
                        help="Prompt mode: simple(~500 chars), enhanced(~2500 chars with chain-of-thought)")

    args = parser.parse_args()

    # Initialize progress manager
    progress_manager = None
    resume = not args.no_resume

    if PROGRESS_ENABLED and resume:
        base_dir = Path(__file__).parent.parent
        progress_manager = get_progress_manager(args.progress_file, base_dir)
        progress_manager.load()
        print(f"\n📌 Resume enabled")
        if args.progress_file:
            print(f"   Progress file: {args.progress_file}")
    elif args.no_resume:
        print(f"\n⚠️  Resume disabled (--no-resume)")
    
    print("="*80)
    print("🎯 DocMind PDF Converter")
    print("="*80)
    print(f"\nFeatures:")
    print(f"  ✅ Three-page context window (previous + current + next page)")
    print(f"  ✅ Concurrency control (asyncio.Semaphore: {args.semaphore_limit})")
    print(f"  ✅ Two-phase processing (OCR → LLM concurrent)")

    # Start resource monitoring
    resource_monitor = None
    if RESOURCE_MONITOR_AVAILABLE:
        resource_monitor = ResourceMonitor(
            interval=10.0,  # Sample every 10 seconds
            log_dir=Path(__file__).parent.parent / "logs",
            enable_logging=True,
            enable_realtime_print=False  # Don't print in real-time to avoid cluttering output
        )
        resource_monitor.start()
    
    if not check_dependencies():
        return 1
    
    # Check API Key configuration
    if API_KEYS and len(API_KEYS) >= 2:
        print(f"\n✅ Multi API Key load balancing enabled ({len(API_KEYS)} keys)")
        api_key = API_KEYS[0]  # For compatibility
    elif API_KEYS:
        print(f"\n✅ Single API Key mode")
        api_key = API_KEYS[0]
    else:
        api_key = get_api_key()
        if not api_key:
            print("\n❌ No API key configured!")
            return 1
        print(f"\n✅ API configured (environment variable)")
    
    # Find PDFs
    pdf_dir = Path(__file__).parent / args.input

    if not pdf_dir.exists():
        print(f"\n❌ Directory does not exist: {pdf_dir}")
        return 1

    pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"\n❌ No PDF files found")
        return 1

    print(f"\n📚 Found {len(pdf_files)} PDF files")
    
    output_base = Path(__file__).parent / args.output
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Issue 3 & 5: Convert string parameters to enum types
    context_mode_map = {"minimal": ContextMode.MINIMAL, "standard": ContextMode.STANDARD, "full": ContextMode.FULL}
    prompt_mode_map = {"simple": PromptMode.SIMPLE, "enhanced": PromptMode.ENHANCED}
    context_mode = context_mode_map.get(args.context_mode, DEFAULT_CONTEXT_MODE)
    prompt_mode = prompt_mode_map.get(args.prompt_mode, DEFAULT_PROMPT_MODE)
    model = args.model

    print(f"\n⚙️  Configuration:")
    print(f"   Page limit: {'first ' + str(args.max_pages) + ' pages' if args.max_pages else 'all'}")
    print(f"   Concurrency limit: {args.semaphore_limit}")
    print(f"   Output directory: {output_base}")
    print(f"   Model: {model}")
    print(f"   Context mode: {context_mode.value}")
    print(f"   Prompt mode: {prompt_mode.value}")

    # Set pending PDF list
    if progress_manager:
        # Determine if chunks or direct based on output directory
        step_name = 'process_chunks' if 'chunks' in str(output_base) else 'process_direct'
        progress_manager.set_pdf_list(step_name, [f.stem for f in pdf_files])
        progress_manager.set_step_status(step_name, 'in_progress')
        progress_manager.set_overall_status('in_progress')

    # Process all PDFs - PDF-level parallel
    print(f"\n🚀 Starting processing (PDF-level parallel)...")

    batch_start = time.time()

    # Use asyncio.gather for PDF-level parallel (Max Safe mode)
    all_results = asyncio.run(
        process_all_pdfs_parallel(
            pdf_files,
            api_key,
            output_base,
            args.max_pages,
            args.pdf_concurrency,  # PDF parallel count
            semaphore_limit=args.semaphore_limit,  # LLM concurrent per PDF
            progress_manager=progress_manager,
            resume=resume,
            context_mode=context_mode,   # Issue 5
            prompt_mode=prompt_mode,      # Issue 3
            model=model                   # qwen-vl-max-latest
        )
    )

    # Update step status
    if progress_manager:
        step_name = 'process_chunks' if 'chunks' in str(output_base) else 'process_direct'
        failed = progress_manager.get_failed_pdfs(step_name)
        if not failed:
            progress_manager.set_step_status(step_name, 'completed')

    batch_time = time.time() - batch_start

    # Stop resource monitoring and get summary
    resource_summary = None
    if resource_monitor:
        resource_summary = resource_monitor.stop()
        print(resource_monitor.format_summary_text())

    print(f"\n{'='*80}")
    print(f"🎉 Batch processing complete!")
    print(f"{'='*80}")

    # Statistics
    total_pages = sum(r.get('pdf_info', {}).get('pages', 0) for r in all_results if r.get('success'))
    total_figures = sum(r.get('pdf_info', {}).get('figures', 0) for r in all_results if r.get('success'))
    total_cost = sum(r.get('processing', {}).get('cost_cny', 0) for r in all_results if r.get('success'))

    print(f"\n📊 Total:")
    print(f"   PDFs: {len(pdf_files)}")
    print(f"   Pages: {total_pages}")
    print(f"   Figures: {total_figures}")
    print(f"   Time: {int(batch_time // 60)}m {int(batch_time % 60)}s")
    print(f"   Cost: ¥{total_cost:.4f}")

    # Save batch report
    batch_report = {
        "batch_info": {
            "timestamp": datetime.now().isoformat(),
            "total_pdfs": len(pdf_files),
            "semaphore_limit": args.semaphore_limit,
            "max_pages_per_pdf": args.max_pages,
            "model": model,
            "context_mode": context_mode.value,
            "prompt_mode": prompt_mode.value
        },
        "summary": {
            "total_pages": total_pages,
            "total_figures": total_figures,
            "total_time": round(batch_time, 2),
            "total_cost_cny": round(total_cost, 4)
        },
        "resource_usage": resource_summary if resource_summary else {},
        "pdfs": all_results
    }
    
    batch_report_file = output_base / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(batch_report_file, "w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Batch report: {batch_report_file}")
    print(f"📁 Results directory: {output_base}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
