#!/bin/bash
# DocMind Lite - PDF to Markdown Converter
# Simplified version for easy deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "DocMind Lite - PDF to Markdown Converter"
echo "======================================================================"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/input"
OUTPUT_DIR="$SCRIPT_DIR/output"
SPLIT_DIR="$SCRIPT_DIR/input/split_pdfs"
LOGS_DIR="$SCRIPT_DIR/logs"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
PROGRESS_FILE="$SCRIPT_DIR/progress.json"
FINAL_DELIVERY="$SCRIPT_DIR/final-delivery"

# Concurrency settings (optimized for 6GB RAM)
MAX_PAGES_PER_CHUNK=${MAX_PAGES:-50}
MAX_SIZE_MB=${MAX_SIZE:-50}
SEMAPHORE=${SEMAPHORE:-4}
LLM_CONCURRENT=${LLM_CONCURRENT:-6}

# Command line arguments
RESTART=false
STATUS_ONLY=false
TASK_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --restart)
            RESTART=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --task-name|-t)
            TASK_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo ""
            echo "Usage: ./run.sh [options]"
            echo ""
            echo "Options:"
            echo "  --restart         Force restart (archive old progress)"
            echo "  --status          Show current progress status only"
            echo "  --task-name, -t   Set task name (for reports)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MAX_PAGES         Max pages per chunk (default: 50)"
            echo "  MAX_SIZE          Max MB per chunk (default: 50)"
            echo "  SEMAPHORE         PDF concurrency (default: 4)"
            echo "  LLM_CONCURRENT    LLM calls per PDF (default: 6)"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                           # Normal run"
            echo "  ./run.sh --restart                 # Fresh start"
            echo "  ./run.sh --task-name my_batch      # With task name"
            echo "  SEMAPHORE=2 ./run.sh               # Lower memory usage"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Status only
if [ "$STATUS_ONLY" = true ]; then
    echo ""
    if [ -f "$PROGRESS_FILE" ]; then
        python3 "$SCRIPTS_DIR/progress_manager.py" --status --progress-file "$PROGRESS_FILE"
    else
        echo "No progress file found. Processing has not started yet."
    fi
    exit 0
fi

# Force restart
if [ "$RESTART" = true ]; then
    echo ""
    echo -e "${YELLOW}Warning: Force restart mode${NC}"
    rm -f "$PROGRESS_FILE"
    rm -f "$PROGRESS_FILE.lock"
    rm -rf "$SPLIT_DIR"
    echo "   Done. Progress reset."
fi

# Auto-generate task name if not specified
if [ -z "$TASK_NAME" ]; then
    TASK_NAME="task_$(date +%Y%m%d_%H%M%S)"
fi
echo "Task name: $TASK_NAME"

# Load environment variables from .env file
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo -e "${GREEN}Loaded .env file${NC}"
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please copy .env.example to .env and fill in your API keys"
    exit 1
fi

# Check API Key
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo -e "${RED}Error: DASHSCOPE_API_KEY not set in .env${NC}"
    exit 1
fi
echo -e "${GREEN}DashScope API Key: [configured]${NC}"

# Check OpenAI API Key (required for content filter recovery)
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set in .env${NC}"
    echo "OpenAI API Key is required for recovering content-filtered pages"
    exit 1
fi
echo -e "${GREEN}OpenAI API Key: [configured]${NC}"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Concurrency: SEMAPHORE=$SEMAPHORE, LLM_CONCURRENT=$LLM_CONCURRENT"

# Check resume status
if [ -f "$PROGRESS_FILE" ]; then
    echo ""
    echo -e "${BLUE}Resume mode: Found progress file${NC}"
    python3 "$SCRIPTS_DIR/progress_manager.py" --status --progress-file "$PROGRESS_FILE" 2>/dev/null || true
    echo ""
fi

echo ""

# Create directories
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$SPLIT_DIR" "$LOGS_DIR" "$FINAL_DELIVERY"

# Check input PDFs
PDF_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$PDF_COUNT" -eq 0 ]; then
    echo "Please put PDF files in the input/ directory"
    exit 0
fi

echo "Found $PDF_COUNT PDF file(s)"
echo ""

START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# Step 1: Split large PDFs
echo "======================================================================"
echo "Step 1: Split large PDFs (>$MAX_PAGES_PER_CHUNK pages or >$MAX_SIZE_MB MB)"
echo "======================================================================"

python3 "$SCRIPTS_DIR/split_large_pdfs_smart.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$SPLIT_DIR" \
    --chunk-size "$MAX_PAGES_PER_CHUNK" \
    --max-chunk-size-mb "$MAX_SIZE_MB" \
    --mapping-file "$SPLIT_DIR/split_mapping.json"

echo ""

# Step 2: Process chunk PDFs
echo "======================================================================"
echo "Step 2: Process chunk PDFs"
echo "======================================================================"

CHUNK_COUNT=$(find "$SPLIT_DIR" -maxdepth 1 -name "*.pdf" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$CHUNK_COUNT" -gt 0 ]; then
    echo "Processing $CHUNK_COUNT chunk PDFs..."

    PYTHONUNBUFFERED=1 python3 "$SCRIPTS_DIR/docmind_converter.py" \
        --input "$SPLIT_DIR" \
        --output "$OUTPUT_DIR/chunks" \
        --semaphore-limit "$LLM_CONCURRENT" \
        --pdf-concurrency "$SEMAPHORE" \
        --progress-file "$PROGRESS_FILE" \
        2>&1 | tee "$LOGS_DIR/chunks_processing.log"
else
    echo "No chunk PDFs to process"
fi

echo ""

# Step 3: Process small PDFs (no splitting needed)
echo "======================================================================"
echo "Step 3: Process small PDFs"
echo "======================================================================"

if [ -f "$SPLIT_DIR/split_mapping.json" ]; then
    DIRECT_COUNT=$(python3 -c "
import json
with open('$SPLIT_DIR/split_mapping.json') as f:
    data = json.load(f)
print(len(data.get('direct', [])))
" 2>/dev/null)

    if [ "$DIRECT_COUNT" -gt 0 ]; then
        echo "Processing $DIRECT_COUNT small PDFs..."

        DIRECT_DIR="$SCRIPT_DIR/input/direct_pdfs"
        mkdir -p "$DIRECT_DIR"
        rm -f "$DIRECT_DIR"/*.pdf 2>/dev/null || true

        python3 -c "
import json
import shutil
from pathlib import Path

with open('$SPLIT_DIR/split_mapping.json') as f:
    data = json.load(f)

direct_dir = Path('$DIRECT_DIR')
for item in data.get('direct', []):
    pdf_path = Path(item['pdf_path'])
    if pdf_path.exists():
        dest_path = direct_dir / pdf_path.name
        if dest_path.exists():
            dest_path.unlink()
        shutil.copy(pdf_path, dest_path)
        print(f'Copied: {pdf_path.name}')
"

        PYTHONUNBUFFERED=1 python3 "$SCRIPTS_DIR/docmind_converter.py" \
            --input "$DIRECT_DIR" \
            --output "$OUTPUT_DIR" \
            --semaphore-limit "$LLM_CONCURRENT" \
            --pdf-concurrency "$SEMAPHORE" \
            --progress-file "$PROGRESS_FILE" \
            2>&1 | tee "$LOGS_DIR/direct_processing.log"
    else
        echo "No small PDFs to process directly"
    fi
fi

echo ""

# Step 4: Merge chunk results
echo "======================================================================"
echo "Step 4: Merge chunk results"
echo "======================================================================"

if [ -f "$SPLIT_DIR/split_mapping.json" ] && [ -d "$OUTPUT_DIR/chunks" ]; then
    python3 "$SCRIPTS_DIR/merge_results_full.py" \
        --mapping-file "$SPLIT_DIR/split_mapping.json" \
        --results-dir "$OUTPUT_DIR/chunks" \
        --output-dir "$OUTPUT_DIR"
fi

# Update progress status
if [ -f "$PROGRESS_FILE" ]; then
    python3 -c "
import json
from datetime import datetime

with open('$PROGRESS_FILE', 'r') as f:
    data = json.load(f)

data['status'] = 'completed'
data['completed_at'] = datetime.now().isoformat()
data['steps']['merge'] = {'status': 'completed', 'completed_at': datetime.now().isoformat()}

with open('$PROGRESS_FILE', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
" 2>/dev/null || true
fi

# Step 5: Create final-delivery folder
echo "======================================================================"
echo "Step 5: Create final-delivery folder"
echo "======================================================================"

mkdir -p "$FINAL_DELIVERY"

for pdf_dir in "$OUTPUT_DIR"/*/; do
    if [[ "$(basename "$pdf_dir")" == "chunks" ]]; then
        continue
    fi

    dir_name=$(basename "$pdf_dir")
    short_name=$(echo "$dir_name" | cut -c1-50 | sed 's/ /-/g' | sed 's/[^a-zA-Z0-9-]//g')

    if [ -n "$short_name" ]; then
        md_file=$(find "$pdf_dir" -maxdepth 1 -name "*.md" -type f | head -1)
        if [ -n "$md_file" ]; then
            cp "$md_file" "$FINAL_DELIVERY/$short_name.md"
            echo "  $short_name.md"
        fi

        yaml_file=$(find "$pdf_dir" -maxdepth 1 -name "*_all_figures.yaml" -type f | head -1)
        if [ -z "$yaml_file" ]; then
            yaml_file=$(find "$pdf_dir" -maxdepth 1 -name "*.yaml" -type f ! -name "*.validation.yaml" | head -1)
        fi
        if [ -n "$yaml_file" ]; then
            cp "$yaml_file" "$FINAL_DELIVERY/$short_name.yaml"
            echo "  $short_name.yaml"
        fi
    fi
done

echo ""
echo "Final delivery: $FINAL_DELIVERY"
echo ""

# Step 6: Markdown Post-processing
echo "======================================================================"
echo "Step 6: Markdown Post-processing"
echo "======================================================================"

python3 "$SCRIPTS_DIR/postprocess.py" \
    --input "$FINAL_DELIVERY" \
    --fix-tables \
    --fix-headings \
    --merge-empty-lines \
    --validate-latex

echo ""

# Step 7: Generate Quality Report
echo "======================================================================"
echo "Step 7: Generate Quality Report"
echo "======================================================================"

python3 "$SCRIPTS_DIR/generate_quality_report.py" \
    --input "$FINAL_DELIVERY" \
    --progress "$PROGRESS_FILE" \
    --json

echo ""

# Step 8: Final Delivery Validation
echo "======================================================================"
echo "Step 8: Final Delivery Validation"
echo "======================================================================"

if [ -f "$SPLIT_DIR/split_mapping.json" ]; then
    EXPECTED_COUNT=$(python3 -c "
import json
with open('$SPLIT_DIR/split_mapping.json') as f:
    data = json.load(f)
chunks = data.get('chunks', [])
direct = data.get('direct', [])
original_pdfs = set(c['original_pdf'] for c in chunks)
original_pdfs.update(d['pdf_path'].split('/')[-1] for d in direct)
print(len(original_pdfs))
" 2>/dev/null)

    python3 "$SCRIPTS_DIR/final_delivery_check.py" \
        --delivery-dir "$FINAL_DELIVERY" \
        --expected-count "$EXPECTED_COUNT" \
        --output-report "$FINAL_DELIVERY/VALIDATION_REPORT.json"
else
    python3 "$SCRIPTS_DIR/final_delivery_check.py" \
        --delivery-dir "$FINAL_DELIVERY" \
        --output-report "$FINAL_DELIVERY/VALIDATION_REPORT.json"
fi

echo ""

# Statistics
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo -e "${GREEN}Processing Complete!${NC}"
echo "======================================================================"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "End time: $(date)"
echo ""

echo "Results:"
YAML_COUNT=$(find "$OUTPUT_DIR" -name "*.yaml" 2>/dev/null | wc -l | tr -d ' ')
MD_COUNT=$(find "$OUTPUT_DIR" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')

echo "  - YAML files: $YAML_COUNT"
echo "  - Markdown files: $MD_COUNT"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Final delivery: $FINAL_DELIVERY"

# Step 9: Content Filter Recovery (Auto)
echo "======================================================================"
echo "Step 9: Content Filter Recovery"
echo "======================================================================"

# Check if there are blocked pages that need recovery
BLOCKED_COUNT=$(python3 -c "
import glob
import yaml
from pathlib import Path

chunks_path = Path('$OUTPUT_DIR/chunks')
if not chunks_path.exists():
    chunks_path = Path('$SCRIPT_DIR/input/direct_pdfs').parent.parent / 'output'

count = 0
for vfile in glob.glob(str(chunks_path) + '/*/*.validation.yaml'):
    try:
        with open(vfile, 'r') as f:
            data = yaml.safe_load(f)
        if data and 'failed_pages_detail' in data:
            for item in data.get('failed_pages_detail', []):
                if isinstance(item, dict):
                    error = str(item.get('error', ''))
                    if 'DataInspectionFailed' in error or 'inappropriate' in error:
                        count += 1
    except:
        pass
print(count)
" 2>/dev/null || echo "0")

if [ "$BLOCKED_COUNT" -gt 0 ]; then
    echo "Found $BLOCKED_COUNT blocked pages, auto-recovering..."
    python3 "$SCRIPTS_DIR/final_data_validation.py" \
        --chunks-dir "$OUTPUT_DIR/chunks" \
        --final-delivery "$FINAL_DELIVERY" \
        --concurrency 10 \
        <<< "y" 2>&1 || echo -e "${YELLOW}Warning: Content filter recovery had issues${NC}"
else
    echo "No blocked pages to recover"
fi

echo ""

# Step 10: YAML Data Verification (Auto)
echo "======================================================================"
echo "Step 10: YAML Data Verification"
echo "======================================================================"

python3 "$SCRIPTS_DIR/yaml_data_tools.py" verify \
    --output-dir "$OUTPUT_DIR/chunks" 2>&1 || \
python3 "$SCRIPTS_DIR/yaml_data_tools.py" verify \
    --output-dir "$OUTPUT_DIR" 2>&1 || echo -e "${YELLOW}Warning: YAML verification skipped${NC}"

echo ""

# Step 11: Sync YAML to Final Delivery (Auto)
echo "======================================================================"
echo "Step 11: Sync YAML to Final Delivery"
echo "======================================================================"

python3 "$SCRIPTS_DIR/yaml_data_tools.py" update \
    --chunks-dir "$OUTPUT_DIR/chunks" \
    --final-delivery "$FINAL_DELIVERY" 2>&1 || \
python3 "$SCRIPTS_DIR/yaml_data_tools.py" update \
    --chunks-dir "$OUTPUT_DIR" \
    --final-delivery "$FINAL_DELIVERY" 2>&1 || echo -e "${YELLOW}Warning: YAML sync skipped${NC}"

echo ""

# Step 12: Generate Task Report
echo "======================================================================"
echo "Step 12: Generate Task Report"
echo "======================================================================"

python3 "$SCRIPTS_DIR/generate_report.py" \
    --task-name "$TASK_NAME" \
    --base-dir "$SCRIPT_DIR" || echo -e "${YELLOW}Warning: Report generation failed (non-critical)${NC}"

echo ""
echo "Report directory: $SCRIPT_DIR/reports/$TASK_NAME/"
echo ""
