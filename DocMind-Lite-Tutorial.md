# (Bilingual) DocMind Lite Tutorial

# Workflow Flowchart (工作流程图)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DocMind Lite Workflow                        │
│                        DocMind Lite 工作流程                         │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   FIRST TIME?    │
    │   第一次使用？     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Run ./setup.sh  │  ◄── Only once! (只需一次!)
    │  运行安装脚本      │
    └────────┬─────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────────┐
│                         NORMAL USAGE                                │
│                          日常使用流程                                │
└────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Step 1: Add PDF │
    │  第一步：添加 PDF  │
    │                  │
    │  Copy PDF files  │
    │  to input/ folder│
    │  将 PDF 复制到    │
    │  input/ 文件夹    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Step 2: Run     │
    │  第二步：运行      │
    │                  │
    │   ./run.sh       │
    │                  │
    │  Wait for it...  │
    │  等待处理完成...   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Step 3: Results │
    │  第三步：获取结果  │
    │                  │
    │  Check folder:   │
    │  final-delivery/ │
    │  查看文件夹：      │
    │  final-delivery/ │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │      DONE!       │
    │      完成！       │
    │                  │
    │  *.md  = Text    │
    │  *.yaml = Data   │
    └──────────────────┘


  ┌─────────────────────────────────────────────────────────────────┐
  │                    IF INTERRUPTED (如果中断)                     │
  │                                                                 │
  │   Just run ./run.sh again - it will resume automatically!       │
  │   只需再次运行 ./run.sh - 它会自动恢复！                           │
  └─────────────────────────────────────────────────────────────────┘
```

---

# 1 Introduction (介绍)

## 1.1 What is DocMind Lite? (什么是 DocMind Lite?)

DocMind Lite is a tool that converts PDF files to Markdown format. It uses AI to extract text, tables, figures, and formulas from PDF documents.

DocMind Lite 是一个将 PDF 文件转换为 Markdown 格式的工具。它使用 AI 从 PDF 文档中提取文字、表格、图表和公式。

**Key Features (主要功能):**

- Converts PDF to clean Markdown text
- Extracts figures and tables with metadata
- Supports LaTeX formula conversion
- Auto-resumes if interrupted
- Processes multiple PDFs at once

- 将 PDF 转换为整洁的 Markdown 文本
- 提取图表和表格及其元数据
- 支持 LaTeX 公式转换
- 中断后自动恢复
- 可同时处理多个 PDF

---

# 2 Setup (安装配置)

## 2.1 Open Terminal (打开终端)

Press `Command + Space`, search for **Terminal** or **iTerm**, and open it.

按下 `Command + 空格键`，搜索 **Terminal** 或 **iTerm**，然后打开。

## 2.2 Navigate to DocMind Folder (进入 DocMind 文件夹)

In the terminal, enter the following command to navigate to the DocMind-Lite folder:

在终端中，输入以下命令进入 DocMind-Lite 文件夹：

```bash
cd /path/to/DocMind-Lite
```

**Command Explanation (命令解释):**

- `cd`: **C**hange **D**irectory (切换目录)。This command moves you into a folder.
- `/path/to/DocMind-Lite`: Replace this with the actual path to your DocMind-Lite folder.

- `cd`: 切换目录。这个命令让您进入一个文件夹。
- `/path/to/DocMind-Lite`: 请替换为您的 DocMind-Lite 文件夹的实际路径。

**Tip (提示):** You can drag the DocMind-Lite folder from Finder directly into the terminal window - it will automatically paste the path!

**提示：** 您可以直接将 DocMind-Lite 文件夹从 Finder 拖到终端窗口中 - 它会自动粘贴路径！

## 2.3 Run Setup Script (运行安装脚本)

Run the setup script to install all required dependencies:

运行安装脚本来安装所有必需的依赖：

```bash
./setup.sh
```

**Command Explanation (命令解释):**

- `./`: Means "in the current folder" (当前文件夹)。
- `setup.sh`: The setup script file name.

- `./`: 表示"在当前文件夹"。
- `setup.sh`: 安装脚本的文件名。

The script will automatically:
- Install Homebrew (if not installed)
- Install Poppler (PDF tools)
- Create Python virtual environment
- Install Python dependencies

脚本会自动：
- 安装 Homebrew（如果未安装）
- 安装 Poppler（PDF 工具）
- 创建 Python 虚拟环境
- 安装 Python 依赖

**Note (注意):** During installation, you may be asked to enter your Mac password. When typing, no characters will appear on screen - this is normal. Press Enter when done.

**注意：** 安装过程中可能会要求您输入 Mac 密码。输入时屏幕上不会显示字符 - 这是正常的。输入完成后按回车键。

---

# 3 Usage (使用方法)

## 3.1 Step 1: Add PDF Files (第一步：添加 PDF 文件)

Copy your PDF files into the `input` folder.

将您的 PDF 文件复制到 `input` 文件夹中。

**Method 1: Using Finder (方法1：使用 Finder)**

1. Open the DocMind-Lite folder in Finder
2. Open the `input` folder
3. Drag and drop your PDF files into this folder

1. 在 Finder 中打开 DocMind-Lite 文件夹
2. 打开 `input` 文件夹
3. 将您的 PDF 文件拖放到此文件夹中

**Method 2: Using Terminal (方法2：使用终端)**

```bash
cp /path/to/your-file.pdf input/
```

**Command Explanation (命令解释):**

- `cp`: **C**o**p**y (复制)。Copies a file to another location.
- `/path/to/your-file.pdf`: The source file path (replace with your actual file).
- `input/`: The destination folder.

- `cp`: 复制。将文件复制到另一个位置。
- `/path/to/your-file.pdf`: 源文件路径（请替换为您的实际文件）。
- `input/`: 目标文件夹。

## 3.2 Step 2: Run Conversion (第二步：运行转换)

In the terminal, make sure you are in the DocMind-Lite folder, then run:

在终端中，确保您在 DocMind-Lite 文件夹内，然后运行：

```bash
./run.sh
```

**Command Explanation (命令解释):**

- `./run.sh`: Runs the main conversion script.

- `./run.sh`: 运行主转换脚本。

The script will:
1. Split large PDFs into smaller chunks (if needed)
2. Convert each page using AI
3. Extract figures and tables
4. Merge results into final output
5. Generate quality report

脚本会：
1. 将大型 PDF 分割成小块（如果需要）
2. 使用 AI 转换每一页
3. 提取图表和表格
4. 合并结果到最终输出
5. 生成质量报告

**Wait for completion (等待完成):** The process may take several minutes depending on PDF size. You will see progress updates in the terminal.

**等待完成：** 根据 PDF 大小，处理过程可能需要几分钟。您会在终端中看到进度更新。

## 3.3 Step 3: Get Results (第三步：获取结果)

After processing completes, find your results in the `final-delivery` folder:

处理完成后，在 `final-delivery` 文件夹中找到您的结果：

| File (文件) | Description (描述) |
|-------------|-------------------|
| `*.md` | Converted Markdown content (转换后的 Markdown 内容) |
| `*.yaml` | Figure and table metadata (图表元数据) |
| `QUALITY_REPORT.md` | Conversion quality report (转换质量报告) |

**To open results in Finder (在 Finder 中打开结果):**

```bash
open final-delivery
```

**Command Explanation (命令解释):**

- `open`: Opens a file or folder in Finder.
- `final-delivery`: The folder containing your converted files.

- `open`: 在 Finder 中打开文件或文件夹。
- `final-delivery`: 包含转换后文件的文件夹。

---

# 4 Common Operations (常用操作)

## 4.1 Check Progress (查看进度)

If you want to check the current processing status:

如果您想查看当前处理状态：

```bash
./run.sh --status
```

## 4.2 Start Fresh (重新开始)

If you want to discard previous progress and start over:

如果您想放弃之前的进度并重新开始：

```bash
./run.sh --restart
```

**Warning (警告):** This will delete all previous progress. Only use this if you want to completely start over.

**警告：** 这会删除所有之前的进度。只有在您想完全重新开始时才使用此命令。

## 4.3 Resume After Interruption (中断后恢复)

If the process was interrupted (e.g., you closed the terminal), simply run `./run.sh` again. It will automatically resume from where it stopped.

如果处理过程被中断（例如，您关闭了终端），只需再次运行 `./run.sh`。它会自动从中断处继续。

```bash
./run.sh
```

---

# 5 Troubleshooting (常见问题)

## 5.1 "Permission denied" Error (权限被拒绝错误)

If you see "permission denied" when running scripts:

如果运行脚本时看到"permission denied"：

```bash
chmod +x setup.sh run.sh
```

**Command Explanation (命令解释):**

- `chmod`: **Ch**ange **Mod**e (更改权限)。Changes file permissions.
- `+x`: Adds e**x**ecute permission (添加执行权限)。Allows the file to run as a script.

- `chmod`: 更改权限。更改文件的权限设置。
- `+x`: 添加执行权限。允许文件作为脚本运行。

## 5.2 "poppler not found" Error (找不到 poppler 错误)

If you see an error about poppler:

如果看到关于 poppler 的错误：

```bash
brew install poppler
```

**Command Explanation (命令解释):**

- `brew`: The Homebrew package manager for Mac.
- `install`: Tells Homebrew to install a program.
- `poppler`: The PDF tools library needed for conversion.

- `brew`: Mac 的 Homebrew 包管理器。
- `install`: 告诉 Homebrew 安装一个程序。
- `poppler`: 转换所需的 PDF 工具库。

## 5.3 Process Seems Stuck (处理似乎卡住了)

If the process appears frozen, check the log files:

如果处理过程似乎卡住了，检查日志文件：

```bash
tail -f logs/direct_processing.log
```

**Command Explanation (命令解释):**

- `tail`: Shows the end of a file.
- `-f`: **F**ollow mode - continuously shows new content as it's added.
- `logs/direct_processing.log`: The log file path.

- `tail`: 显示文件的末尾内容。
- `-f`: 跟随模式 - 持续显示新添加的内容。
- `logs/direct_processing.log`: 日志文件路径。

Press `Ctrl + C` to stop viewing the log.

按 `Ctrl + C` 停止查看日志。

---

# 6 Quick Reference (快速参考)

| Command (命令) | Description (描述) |
|----------------|-------------------|
| `./setup.sh` | Install dependencies (安装依赖) |
| `./run.sh` | Start/resume conversion (开始/恢复转换) |
| `./run.sh --status` | Check progress (查看进度) |
| `./run.sh --restart` | Start fresh (重新开始) |
| `open input` | Open input folder (打开输入文件夹) |
| `open final-delivery` | Open results folder (打开结果文件夹) |

---

# 7 Summary (总结)

**Basic workflow (基本工作流程):**

1. **Setup (安装):** Run `./setup.sh` once
2. **Add files (添加文件):** Put PDFs in `input/` folder
3. **Convert (转换):** Run `./run.sh`
4. **Get results (获取结果):** Check `final-delivery/` folder

1. **安装：** 运行一次 `./setup.sh`
2. **添加文件：** 将 PDF 放入 `input/` 文件夹
3. **转换：** 运行 `./run.sh`
4. **获取结果：** 查看 `final-delivery/` 文件夹

That's it! If you have any questions, please contact the development team.

就这些！如果您有任何问题，请联系开发团队。
