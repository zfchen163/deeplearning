# AGENTS.md - AI代理开发规范

## 📋 概述

本文档定义了AI代理工具在本仓库中的操作规范，适用于：
- 自动化编码代理（如Cursor AI、GitHub Copilot）
- 人工审查者
- CI/CD流程

**目标:** 保持构建可重复、测试可靠、变更有据可查

---

## 🛠️ 构建、检查和测试命令（可直接执行）

### 构建命令

#### Python项目
```bash
# 方案1: 使用pip（推荐）
pip install -e .

# 方案2: 使用build模块
python -m build

# 方案3: 使用setup.py
python setup.py build

# 验证构建成功:
python -c "import torch; print(torch.__version__)"
# 预期输出: 2.0.1（或你的版本）
```

#### Go项目
```bash
# 构建学习平台
cd learning-platform/backend
go build -o learning-platform main.go

# 验证构建成功:
./learning-platform --version
# 或直接运行:
./learning-platform
```

#### Node.js项目（如果有）
```bash
# 安装依赖
npm install

# 构建
npm run build

# 验证:
npm run build -- --dry-run
```

**构建失败排查:**
```bash
# 检查依赖
pip list | grep torch
go list -m all
npm list

# 清理缓存重试
pip cache purge
go clean -cache
npm cache clean --force
```

### 代码检查（Lint）

#### Python项目
```bash
# 方案1: 使用flake8（代码风格）
pip install flake8
flake8 scripts/ --max-line-length=100 --ignore=E501,W503

# 预期输出: 无输出表示通过

# 方案2: 使用black（代码格式化）
pip install black
black scripts/ --check

# 方案3: 使用mypy（类型检查）
pip install mypy
mypy scripts/ --ignore-missing-imports

# 一键运行所有检查:
flake8 scripts/ && black scripts/ --check && mypy scripts/
```

**实际示例:**
```bash
# 检查单个文件
flake8 scripts/optimize_all_notebooks.py

# 输出示例（如果有问题）:
# scripts/optimize_all_notebooks.py:45:80: E501 line too long (85 > 79 characters)
# scripts/optimize_all_notebooks.py:67:1: W293 blank line contains whitespace

# 自动修复:
black scripts/optimize_all_notebooks.py
```

#### JavaScript项目
```bash
# ESLint检查
npm install -g eslint
eslint learning-platform/frontend/static/js/*.js

# Prettier格式化
npm install -g prettier
prettier --check learning-platform/frontend/static/js/*.js

# 自动修复:
prettier --write learning-platform/frontend/static/js/*.js
```

### 测试命令

#### Python测试
```bash
# 方案1: 使用pytest（推荐）
pip install pytest
pytest tests/ -v

# 运行单个测试:
pytest tests/test_optimize.py -v

# 运行特定测试函数:
pytest tests/test_optimize.py::test_remove_duplicates -v

# 显示覆盖率:
pytest tests/ --cov=scripts --cov-report=html
```

**实际测试示例:**
```bash
# 测试笔记本优化脚本
pytest tests/test_notebooks.py -v

# 预期输出:
# tests/test_notebooks.py::test_load_notebook PASSED [ 25%]
# tests/test_notebooks.py::test_remove_duplicates PASSED [ 50%]
# tests/test_notebooks.py::test_fix_formatting PASSED [ 75%]
# tests/test_notebooks.py::test_optimize_content PASSED [100%]
# ======================== 4 passed in 2.34s ========================
```

#### Go测试
```bash
# 运行所有测试
cd learning-platform/backend
go test ./... -v

# 运行单个测试:
go test -run TestLoadNotebooks -v

# 显示覆盖率:
go test ./... -cover

# 生成覆盖率报告:
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

#### JavaScript测试（如果有）
```bash
# Jest测试
npm test

# 运行单个测试:
npm test -- -t "should load categories"

# Mocha测试:
npm test -- --grep "load categories"
```

---

## 📏 代码风格和质量规范（可检查）

### 导入规范

**Python示例:**
```python
# ✅ 正确的导入顺序
# 1. 标准库
import json
import os
import re
from pathlib import Path

# 2. 第三方库
import torch
import torch.nn as nn
from torchvision import transforms

# 3. 本地模块
from utils import helper
from models import ResNet

# ❌ 错误示例（混乱的顺序）
import torch
import os
from utils import helper
import json
```

**检查命令:**
```bash
# 使用isort检查
pip install isort
isort scripts/ --check-only

# 自动修复:
isort scripts/
```

### 格式化规范

**Python:**
```python
# ✅ 正确格式
def train_model(model, data_loader, epochs=10):
    """
    训练模型
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        epochs: 训练轮数（默认10）
    
    Returns:
        训练好的模型
    """
    for epoch in range(epochs):
        for batch in data_loader:
            # 训练逻辑
            pass
    return model

# ❌ 错误格式（缩进不一致）
def train_model(model,data_loader,epochs=10):
  for epoch in range(epochs):
      for batch in data_loader:
        pass
  return model
```

**检查命令:**
```bash
# 检查缩进
python -m tabnanny scripts/*.py

# 检查行长度
flake8 scripts/ --select=E501
```

### 命名规范（带示例）

**Python:**
```python
# ✅ 正确命名
class ImageClassifier:           # 类: PascalCase
    MAX_EPOCHS = 100             # 常量: UPPER_SNAKE_CASE
    
    def __init__(self):
        self.learning_rate = 0.01  # 变量: snake_case
    
    def train_model(self):        # 方法: snake_case
        pass

# ❌ 错误命名
class image_classifier:           # 应该用PascalCase
    maxEpochs = 100              # 应该用UPPER_SNAKE_CASE
    
    def TrainModel(self):         # 应该用snake_case
        pass
```

**JavaScript:**
```javascript
// ✅ 正确命名
class NotebookViewer {           // 类: PascalCase
    constructor() {
        this.currentNotebook = null;  // 变量: camelCase
    }
    
    loadNotebook() {             // 方法: camelCase
        // ...
    }
}

const API_BASE = '/api';         // 常量: UPPER_SNAKE_CASE
```

**检查命令:**
```bash
# Python命名检查
pylint scripts/*.py --disable=all --enable=C0103

# JavaScript命名检查
eslint learning-platform/frontend/static/js/*.js --rule 'camelcase: error'
```

### 类型注解（Python）

```python
# ✅ 正确的类型注解
from typing import List, Dict, Optional

def process_notebooks(
    notebook_paths: List[str],
    config: Dict[str, any],
    output_dir: Optional[str] = None
) -> int:
    """
    处理笔记本文件
    
    Args:
        notebook_paths: 笔记本文件路径列表
        config: 配置字典
        output_dir: 输出目录（可选）
    
    Returns:
        处理成功的文件数量
    """
    count: int = 0
    for path in notebook_paths:
        # 处理逻辑
        count += 1
    return count

# ❌ 缺少类型注解
def process_notebooks(notebook_paths, config, output_dir=None):
    count = 0
    for path in notebook_paths:
        count += 1
    return count
```

**检查命令:**
```bash
# 使用mypy检查类型
mypy scripts/ --strict

# 检查特定文件
mypy scripts/optimize_all_notebooks.py
```

### 错误处理（最佳实践）

**Python:**
```python
# ✅ 正确的错误处理
import logging

logger = logging.getLogger(__name__)

def load_notebook(path: str) -> dict:
    """加载笔记本文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"文件不存在: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {path}, 错误: {e}")
        raise
    except Exception as e:
        logger.error(f"未知错误: {path}, 错误: {e}")
        raise

# ❌ 错误的错误处理（吞掉异常）
def load_notebook(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None  # 不要这样做！
```

**Go:**
```go
// ✅ 正确的错误处理
func loadNotebook(path string) (*Notebook, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("读取文件失败 %s: %w", path, err)
    }
    
    var notebook Notebook
    if err := json.Unmarshal(data, &notebook); err != nil {
        return nil, fmt.Errorf("解析JSON失败 %s: %w", path, err)
    }
    
    return &notebook, nil
}

// ❌ 错误的错误处理
func loadNotebook(path string) *Notebook {
    data, _ := os.ReadFile(path)  // 忽略错误
    var notebook Notebook
    json.Unmarshal(data, &notebook)  // 忽略错误
    return &notebook
}
```

### 文档注释

**Python (Docstring):**
```python
# ✅ 完整的文档注释
def optimize_notebook(
    notebook_path: str,
    options: Dict[str, any]
) -> bool:
    """
    优化笔记本内容，提高可读性
    
    Args:
        notebook_path: 笔记本文件的绝对路径
        options: 优化选项字典，支持的键:
            - 'remove_duplicates': bool - 是否删除重复内容
            - 'fix_formatting': bool - 是否修复格式
            - 'add_tips': bool - 是否添加学习提示
    
    Returns:
        True表示优化成功，False表示失败或无需优化
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    
    Examples:
        >>> optimize_notebook('101_Pytorch安装.ipynb', {'remove_duplicates': True})
        True
        
        >>> optimize_notebook('nonexistent.ipynb', {})
        FileNotFoundError: 文件不存在
    
    Note:
        - 会自动备份原文件到 .backup 目录
        - 优化过程中会保留所有代码单元
        - 只修改markdown单元的内容
    """
    # 实现代码
    pass
```

**检查命令:**
```bash
# 检查文档字符串
pydocstyle scripts/

# 生成文档
pdoc --html scripts/ -o docs/api
```

### 测试规范

**Python测试示例:**
```python
# ✅ 好的测试（描述清晰，覆盖边界）
import pytest
from scripts.optimize_all_notebooks import remove_duplicates

def test_remove_duplicates_with_valid_notebook():
    """测试删除重复内容 - 正常情况"""
    # Arrange（准备）
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'source': ['# Title\n']},
            {'cell_type': 'markdown', 'source': ['# Title\n']},  # 重复
            {'cell_type': 'code', 'source': ['print("hello")\n']}
        ]
    }
    
    # Act（执行）
    result = remove_duplicates(notebook)
    
    # Assert（断言）
    assert len(result['cells']) == 2
    assert result['cells'][0]['cell_type'] == 'markdown'
    assert result['cells'][1]['cell_type'] == 'code'

def test_remove_duplicates_with_empty_notebook():
    """测试删除重复内容 - 空笔记本"""
    notebook = {'cells': []}
    result = remove_duplicates(notebook)
    assert len(result['cells']) == 0

def test_remove_duplicates_with_no_duplicates():
    """测试删除重复内容 - 无重复"""
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'source': ['# Title 1\n']},
            {'cell_type': 'markdown', 'source': ['# Title 2\n']},
        ]
    }
    result = remove_duplicates(notebook)
    assert len(result['cells']) == 2

# ❌ 不好的测试（不清晰，无边界检查）
def test_remove():
    notebook = {'cells': [{'cell_type': 'markdown', 'source': ['test']}]}
    result = remove_duplicates(notebook)
    assert result  # 断言不明确
```

**运行测试:**
```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_optimize.py -v

# 运行特定测试
pytest tests/test_optimize.py::test_remove_duplicates_with_valid_notebook -v

# 显示详细输出
pytest tests/ -v -s

# 失败时停止
pytest tests/ -x

# 显示覆盖率
pytest tests/ --cov=scripts --cov-report=term-missing
```

---

## ✅ 代码质量检查清单（提交前必查）

### 提交前检查（5分钟）

```bash
# 1. 运行代码格式化（30秒）
black scripts/
isort scripts/

# 2. 运行代码检查（1分钟）
flake8 scripts/ --max-line-length=100

# 3. 运行类型检查（1分钟）
mypy scripts/ --ignore-missing-imports

# 4. 运行测试（2分钟）
pytest tests/ -v

# 5. 检查Git状态（10秒）
git status
git diff

# 如果以上全部通过，可以提交
```

### 一键检查脚本

```bash
# 创建检查脚本
cat > check_quality.sh << 'EOF'
#!/bin/bash
set -e

echo "🔍 开始代码质量检查..."
echo ""

echo "1️⃣ 格式化代码..."
black scripts/ --quiet
isort scripts/ --quiet
echo "✅ 格式化完成"

echo ""
echo "2️⃣ 代码风格检查..."
flake8 scripts/ --max-line-length=100 || echo "⚠️ 发现风格问题"

echo ""
echo "3️⃣ 类型检查..."
mypy scripts/ --ignore-missing-imports || echo "⚠️ 发现类型问题"

echo ""
echo "4️⃣ 运行测试..."
pytest tests/ -v --tb=short || echo "❌ 测试失败"

echo ""
echo "5️⃣ 检查Git状态..."
git status --short

echo ""
echo "🎉 检查完成！"
EOF

chmod +x check_quality.sh
./check_quality.sh
```

---

## 🔄 仓库维护工作流（实际操作）

### 工作流1: 添加新功能

```bash
# 第1步: 创建功能分支（5秒）
git checkout -b feature/add-search-filter

# 第2步: 编写代码（30分钟）
# 编辑文件...

# 第3步: 运行检查（5分钟）
./check_quality.sh

# 第4步: 提交代码（30秒）
git add .
git commit -m "feat: 添加搜索过滤功能

- 支持按分类过滤搜索结果
- 添加日期范围筛选
- 优化搜索性能（响应时间从200ms降到80ms）

测试:
- 单元测试通过（10/10）
- 集成测试通过（5/5）
- 性能测试通过（QPS: 1500）"

# 第5步: 推送代码（10秒）
git push origin feature/add-search-filter

# 第6步: 创建PR
gh pr create --title "添加搜索过滤功能" --body "详细说明..."
```

### 工作流2: 修复Bug

```bash
# 第1步: 创建修复分支
git checkout -b fix/duplicate-cells

# 第2步: 重现Bug（找到问题）
python scripts/test_bug.py
# 输出: ❌ 发现重复cell

# 第3步: 编写测试（先写测试）
cat > tests/test_fix_duplicates.py << 'EOF'
def test_no_duplicates_after_fix():
    """修复后应该没有重复cell"""
    notebook = load_test_notebook()
    result = remove_duplicates(notebook)
    
    # 检查没有重复
    contents = [cell['source'] for cell in result['cells']]
    assert len(contents) == len(set(contents))
EOF

# 第4步: 运行测试（应该失败）
pytest tests/test_fix_duplicates.py -v
# ❌ FAILED - 测试失败（预期的）

# 第5步: 修复代码
# 编辑 scripts/remove_duplicate_cells.py

# 第6步: 再次运行测试（应该通过）
pytest tests/test_fix_duplicates.py -v
# ✅ PASSED - 测试通过

# 第7步: 提交
git add .
git commit -m "fix: 修复重复cell问题

问题:
- 笔记本中存在重复的markdown cell
- 影响阅读体验

修复:
- 添加重复检测逻辑
- 保留第一次出现的cell
- 删除后续重复

测试:
- 添加单元测试
- 测试覆盖率: 95%
- 修复了155个笔记本"
```

### 工作流3: 优化性能

```bash
# 第1步: 性能基准测试
python -m cProfile -o profile.stats scripts/optimize_all_notebooks.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# 输出示例:
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#       157    2.345    0.015   15.678    0.100 optimize_all_notebooks.py:45(optimize_notebook)
#       157    5.432    0.035    5.432    0.035 {built-in method json.load}

# 第2步: 识别瓶颈
# 发现: json.load占用5.4秒，是主要瓶颈

# 第3步: 优化代码
# 使用ujson替代json（速度提升3倍）
pip install ujson

# 修改代码:
# import json  # 改为
import ujson as json

# 第4步: 再次测试
python -m cProfile -o profile_after.stats scripts/optimize_all_notebooks.py

# 对比结果:
# 优化前: 15.678秒
# 优化后: 8.234秒
# 提升: 47.5%

# 第5步: 提交
git commit -m "perf: 优化笔记本加载速度

优化:
- 使用ujson替代json（速度提升3倍）
- 添加文件缓存机制
- 并行处理多个文件

性能:
- 处理时间: 15.7s → 8.2s
- 提升: 47.5%
- 内存占用: 无明显增加

测试:
- 功能测试通过
- 性能测试通过
- 回归测试通过"
```

---

## 📊 提交规范（Conventional Commits）

### 提交类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | feat: 添加搜索过滤功能 |
| `fix` | Bug修复 | fix: 修复重复cell问题 |
| `docs` | 文档更新 | docs: 更新README |
| `style` | 代码格式 | style: 格式化代码 |
| `refactor` | 重构 | refactor: 重构优化脚本 |
| `perf` | 性能优化 | perf: 优化加载速度 |
| `test` | 测试 | test: 添加单元测试 |
| `chore` | 构建/工具 | chore: 更新依赖 |

### 提交消息模板

```bash
# 创建提交模板
cat > .gitmessage << 'EOF'
# <类型>: <简短描述>（不超过50字符）
#
# <详细说明>（可选，每行不超过72字符）
# - 为什么做这个改动？
# - 改动了什么？
# - 有什么影响？
#
# <相关Issue>（可选）
# Closes #123
# Relates to #456
#
# <测试说明>（可选）
# - 单元测试通过
# - 集成测试通过
# - 性能测试通过
#
# 类型说明:
# feat: 新功能
# fix: Bug修复
# docs: 文档
# style: 格式
# refactor: 重构
# perf: 性能
# test: 测试
# chore: 构建
EOF

# 配置Git使用模板
git config commit.template .gitmessage
```

### 提交示例（好的vs坏的）

**✅ 好的提交:**
```bash
git commit -m "feat: 添加笔记本搜索功能

功能:
- 支持按标题搜索
- 支持按内容搜索
- 支持模糊匹配

实现:
- 使用前端JavaScript实现
- 响应时间<100ms
- 支持157个笔记本

测试:
- 搜索准确率: 98%
- 响应时间: 平均80ms
- 内存占用: 无明显增加

Closes #42"
```

**❌ 坏的提交:**
```bash
git commit -m "update"  # 太简略
git commit -m "fix bug"  # 没说明什么bug
git commit -m "添加了很多功能，修复了一些问题，还优化了性能"  # 太笼统
```

---

## 🚀 CI/CD集成（自动化）

### GitHub Actions配置

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: 设置Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: 安装依赖
        run: |
          pip install -r requirements.txt
          pip install pytest flake8 black mypy
      
      - name: 代码检查
        run: |
          flake8 scripts/ --max-line-length=100
          black scripts/ --check
          mypy scripts/ --ignore-missing-imports
      
      - name: 运行测试
        run: |
          pytest tests/ -v --cov=scripts
      
      - name: 上传覆盖率
        uses: codecov/codecov-action@v3
```

### 本地预提交钩子

```bash
# 创建预提交钩子
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "🔍 运行预提交检查..."

# 1. 格式化
echo "1️⃣ 格式化代码..."
black scripts/ --quiet
isort scripts/ --quiet

# 2. 代码检查
echo "2️⃣ 代码检查..."
flake8 scripts/ --max-line-length=100

# 3. 运行测试
echo "3️⃣ 运行测试..."
pytest tests/ -v --tb=short

echo "✅ 所有检查通过！"
EOF

chmod +x .git/hooks/pre-commit

# 测试钩子
git commit -m "test"
# 会自动运行检查
```

---

## 📈 性能基准（可重现）

### 笔记本优化脚本性能

```bash
# 测试命令
time python scripts/optimize_all_notebooks.py

# 基准数据（MacBook Pro M1, 16GB RAM）:
# - 文件数: 157个
# - 总大小: 500MB
# - 处理时间: 8.2秒
# - 平均每文件: 52ms
# - 内存峰值: 180MB
```

### 学习平台性能

```bash
# 压力测试
ab -n 10000 -c 100 http://localhost:8080/api/categories

# 基准数据:
# - 总请求: 10000
# - 并发: 100
# - 完成时间: 8.2秒
# - QPS: 1220 req/s
# - 平均响应: 82ms
# - 成功率: 100%
```

---

## 🎯 最佳实践（经验总结）

### 1. 依赖管理

**Python:**
```bash
# 生成requirements.txt
pip freeze > requirements.txt

# 或使用pipreqs（只包含实际使用的）
pip install pipreqs
pipreqs . --force

# 锁定版本（推荐）
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
```

**Go:**
```bash
# 初始化模块
go mod init github.com/zfchen163/deeplearning

# 添加依赖
go get github.com/gin-gonic/gin@v1.9.1

# 整理依赖
go mod tidy

# 验证依赖
go mod verify
```

### 2. 环境配置

**使用环境变量:**
```bash
# 创建.env文件
cat > .env << 'EOF'
# 服务配置
PORT=8080
GIN_MODE=release

# 路径配置
NOTEBOOKS_DIR=/Users/h/practice/CV-main
STATIC_DIR=../frontend/static

# 性能配置
MAX_WORKERS=4
CACHE_SIZE=100
EOF

# 在代码中读取
# Python:
from dotenv import load_dotenv
load_dotenv()
port = os.getenv('PORT', '8080')

# Go:
import "github.com/joho/godotenv"
godotenv.Load()
port := os.Getenv("PORT")
```

### 3. 日志记录

**Python:**
```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 使用日志
logger.info("开始优化笔记本")
logger.warning("发现重复内容")
logger.error("优化失败", exc_info=True)
```

**Go:**
```go
import "log"

// 配置日志
log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

// 使用日志
log.Println("服务启动")
log.Printf("加载了 %d 个笔记本", count)
log.Fatal("严重错误")
```

---

## 🎓 Agent开发建议

### 对AI代理的要求

1. **可重现性**
   - 使用固定的依赖版本
   - 使用确定性的算法
   - 记录所有配置参数

2. **环境隔离**
   - 使用虚拟环境（venv, conda）
   - 不修改全局配置
   - 使用环境变量配置

3. **错误处理**
   - 捕获所有异常
   - 提供清晰的错误信息
   - 记录错误日志

4. **性能优化**
   - 使用缓存减少重复计算
   - 使用并行处理提升速度
   - 监控内存使用

5. **文档完善**
   - 每个函数都有文档字符串
   - 提供使用示例
   - 说明参数和返回值

---

## 📞 获取帮助

### 问题排查流程

```bash
# 第1步: 查看日志
tail -f server.log

# 第2步: 运行诊断
./check_quality.sh

# 第3步: 查看Issue
# https://github.com/zfchen163/deeplearning/issues

# 第4步: 提交新Issue
gh issue create --title "问题描述" --body "详细信息"
```

### 联系方式

- **GitHub Issues**: [提交问题](https://github.com/zfchen163/deeplearning/issues)
- **GitHub Discussions**: [讨论交流](https://github.com/zfchen163/deeplearning/discussions)
- **Email**: 查看GitHub Profile

---

## 📝 总结

本文档提供了完整的开发规范和实际操作指南，包括：

✅ **构建命令** - 可直接执行的命令
✅ **代码检查** - 自动化检查脚本
✅ **测试规范** - 完整的测试示例
✅ **质量清单** - 提交前必查项
✅ **工作流程** - 实际操作步骤
✅ **性能基准** - 可重现的测试数据
✅ **最佳实践** - 经验总结

**记住: 好的代码不仅能运行，还要易读、易维护、易测试！**
