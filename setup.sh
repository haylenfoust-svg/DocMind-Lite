#!/bin/bash

# ====================================
# DocMind Lite - One-Click Setup Script
# For macOS / Linux
# Version: 2.0 - December 2024
# ====================================

set -e

# Auto-fix permissions
chmod +x "$0" run.sh 2>/dev/null || true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

cat << 'BANNER'
======================================================================
DocMind Lite - One-Click Setup
======================================================================

This will install:
- Xcode Command Line Tools (macOS)
- Homebrew (macOS)
- Poppler (PDF tools)
- Python 3 & virtual environment
- All Python dependencies
- Configure API keys

BANNER

read -p "Continue with installation? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
fi

print_info "Detected OS: $OS"
echo ""

# ====================================
# Step 1: Xcode Command Line Tools (macOS only)
# ====================================
echo "======================================================================"
echo "Step 1: Xcode Command Line Tools"
echo "======================================================================"

if [[ "$OS" == "macos" ]]; then
    if xcode-select -p &> /dev/null; then
        print_status "Xcode Command Line Tools already installed"
    else
        print_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo ""
        print_warning "Please complete the Xcode installation in the popup window."
        read -p "Press Enter after installation completes..." -r
    fi
else
    print_info "Skipping (Linux detected)"
fi

echo ""

# ====================================
# Step 2: Homebrew (macOS only)
# ====================================
echo "======================================================================"
echo "Step 2: Package Manager"
echo "======================================================================"

if [[ "$OS" == "macos" ]]; then
    if command -v brew &> /dev/null; then
        print_status "Homebrew already installed"
        brew update 2>/dev/null || print_warning "Homebrew update skipped"
    else
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add to PATH for Apple Silicon
        if [[ -f /opt/homebrew/bin/brew ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        print_status "Homebrew installed"
    fi
elif [[ "$OS" == "linux" ]]; then
    if command -v apt &> /dev/null; then
        print_status "APT package manager detected"
        sudo apt update -qq
    elif command -v yum &> /dev/null; then
        print_status "YUM package manager detected"
    elif command -v dnf &> /dev/null; then
        print_status "DNF package manager detected"
    else
        print_error "No supported package manager found"
        exit 1
    fi
fi

echo ""

# ====================================
# Step 3: Install Poppler
# ====================================
echo "======================================================================"
echo "Step 3: Install Poppler (PDF tools)"
echo "======================================================================"

if command -v pdftoppm &> /dev/null; then
    POPPLER_VERSION=$(pdftoppm -v 2>&1 | head -1)
    print_status "Poppler already installed: $POPPLER_VERSION"
else
    print_info "Installing Poppler..."
    if [[ "$OS" == "macos" ]]; then
        brew install poppler 2>/dev/null || print_warning "Poppler install issue (may already exist)"
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt &> /dev/null; then
            sudo apt install -y poppler-utils
        elif command -v yum &> /dev/null; then
            sudo yum install -y poppler-utils
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y poppler-utils
        fi
    fi

    if command -v pdftoppm &> /dev/null; then
        print_status "Poppler installed successfully"
    else
        print_error "Poppler installation failed. Please install manually."
        exit 1
    fi
fi

echo ""

# ====================================
# Step 4: Check/Install Python & pip
# ====================================
echo "======================================================================"
echo "Step 4: Check Python & pip"
echo "======================================================================"

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

if [[ -z "$PYTHON_CMD" ]]; then
    print_warning "Python not found. Installing..."
    if [[ "$OS" == "macos" ]]; then
        brew install python@3.12 2>/dev/null || brew install python
        PYTHON_CMD="python3"
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt &> /dev/null; then
            sudo apt install -y python3 python3-pip python3-venv
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip
        fi
        PYTHON_CMD="python3"
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
print_status "Python: $PYTHON_VERSION"

# Check/Install pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_warning "pip not found. Installing..."
    if [[ "$OS" == "macos" ]]; then
        # macOS: pip comes with Homebrew Python, try reinstall
        brew reinstall python@3.12 2>/dev/null || brew reinstall python 2>/dev/null || true
        # Fallback: use ensurepip
        $PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || true
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt &> /dev/null; then
            sudo apt install -y python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3-pip
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3-pip
        fi
        # Fallback: use get-pip.py
        if ! $PYTHON_CMD -m pip --version &> /dev/null; then
            curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
            $PYTHON_CMD /tmp/get-pip.py --user
            rm -f /tmp/get-pip.py
        fi
    fi
fi

if $PYTHON_CMD -m pip --version &> /dev/null; then
    PIP_VERSION=$($PYTHON_CMD -m pip --version 2>&1)
    print_status "pip: $PIP_VERSION"
else
    print_error "pip installation failed. Please install manually."
    exit 1
fi

echo ""

# ====================================
# Step 5: Create Virtual Environment
# ====================================
echo "======================================================================"
echo "Step 5: Create Virtual Environment"
echo "======================================================================"

if [[ -d "venv" ]]; then
    print_status "Virtual environment already exists"
else
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created"
fi

# Activate venv
source venv/bin/activate
print_status "Virtual environment activated"

echo ""

# ====================================
# Step 6: Install Python Dependencies
# ====================================
echo "======================================================================"
echo "Step 6: Install Python Dependencies"
echo "======================================================================"

print_info "Upgrading pip..."
pip install --upgrade pip -q

print_info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt -q

print_status "All Python dependencies installed"

echo ""

# ====================================
# Step 7: Configure API Keys
# ====================================
echo "======================================================================"
echo "Step 7: Configure API Keys"
echo "======================================================================"

# Backup existing .env if exists
if [[ -f ".env" ]]; then
    # Check if keys are placeholder values
    if grep -q "your-dashscope-api-key-here" .env 2>/dev/null || grep -q "your-openai-api-key-here" .env 2>/dev/null; then
        print_warning ".env exists but API keys not configured"
        NEED_CONFIG=true
    else
        print_status ".env file exists with API keys configured"
        NEED_CONFIG=false
    fi
else
    print_info "Creating .env from template..."
    cp .env.example .env
    NEED_CONFIG=true
fi

if [[ "$NEED_CONFIG" == "true" ]]; then
    echo ""
    print_info "Please enter your API keys:"
    echo ""
    echo "Get DashScope key: https://dashscope.console.aliyun.com/"
    echo "Get OpenAI key: https://platform.openai.com/api-keys"
    echo ""

    # Ask for DashScope API Key
    read -p "Enter DASHSCOPE_API_KEY (Aliyun, required): " DASHSCOPE_KEY
    if [[ -n "$DASHSCOPE_KEY" ]]; then
        if [[ "$OS" == "macos" ]]; then
            sed -i '' "s/DASHSCOPE_API_KEY=.*/DASHSCOPE_API_KEY=$DASHSCOPE_KEY/" .env
        else
            sed -i "s/DASHSCOPE_API_KEY=.*/DASHSCOPE_API_KEY=$DASHSCOPE_KEY/" .env
        fi
        print_status "DASHSCOPE_API_KEY configured"
    else
        print_warning "DASHSCOPE_API_KEY skipped (required for conversion)"
    fi

    # Ask for OpenAI API Key
    read -p "Enter OPENAI_API_KEY (OpenAI, optional for recovery): " OPENAI_KEY
    if [[ -n "$OPENAI_KEY" ]]; then
        if [[ "$OS" == "macos" ]]; then
            sed -i '' "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" .env
        else
            sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" .env
        fi
        print_status "OPENAI_API_KEY configured"
    else
        print_warning "OPENAI_API_KEY skipped (needed for content filter recovery)"
    fi
fi

echo ""

# ====================================
# Step 8: Create Directories
# ====================================
echo "======================================================================"
echo "Step 8: Create Directories"
echo "======================================================================"

mkdir -p input output final-delivery logs reports
print_status "Directories created: input/ output/ final-delivery/ logs/ reports/"

echo ""

# ====================================
# Step 9: Set Permissions
# ====================================
echo "======================================================================"
echo "Step 9: Set Permissions"
echo "======================================================================"

chmod +x run.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
print_status "Scripts are executable"

echo ""

# ====================================
# Verification
# ====================================
echo "======================================================================"
echo "Verification"
echo "======================================================================"

PASS=0
FAIL=0

echo -n "  poppler:  "
if command -v pdftoppm &> /dev/null; then
    print_status "OK"
    ((PASS++))
else
    print_error "NOT FOUND"
    ((FAIL++))
fi

echo -n "  python:   "
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    print_status "OK"
    ((PASS++))
else
    print_error "NOT FOUND"
    ((FAIL++))
fi

echo -n "  venv:     "
if [[ -d "venv" ]]; then
    print_status "OK"
    ((PASS++))
else
    print_error "NOT FOUND"
    ((FAIL++))
fi

echo -n "  .env:     "
if [[ -f ".env" ]]; then
    print_status "OK"
    ((PASS++))
else
    print_error "NOT FOUND"
    ((FAIL++))
fi

echo -n "  API keys: "
if [[ -f ".env" ]] && ! grep -q "your-dashscope-api-key-here" .env 2>/dev/null; then
    print_status "OK"
    ((PASS++))
else
    print_warning "Not configured"
fi

echo ""

# ====================================
# Summary
# ====================================
if [[ $FAIL -eq 0 ]]; then
    cat << 'SUCCESS'
======================================================================
Setup Complete!
======================================================================

Next steps:

  1. Put your PDF files in the input/ folder:
     cp your-file.pdf input/

  2. Activate virtual environment:
     source venv/bin/activate

  3. Run the converter:
     ./run.sh

SUCCESS
else
    print_error "Setup completed with $FAIL errors. Please fix before running."
fi

echo ""
