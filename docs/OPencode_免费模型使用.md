# OpenCode 免费模型使用指南 🆓

## 🎯 快速开始（无需 API Key）

### 立即使用免费模型

```bash
# 方法1：使用完整路径
~/.opencode/bin/opencode -m opencode/gpt-5-nano

# 方法2：如果 PATH 已设置
opencode -m opencode/gpt-5-nano
```

---

## ✅ 可用的免费模型

根据 [OpenCode 官方文档](https://opencode.ai/docs/zen/)，OpenCode 提供以下**完全免费**的模型：

| 模型 | 定价 | 说明 | 推荐度 |
|------|------|------|--------|
| `opencode/gpt-5-nano` | **完全免费** | GPT-5 Nano 版本，专为代码生成优化 | ⭐⭐⭐⭐⭐ |
| `opencode/big-pickle` | **完全免费** | Stealth 模型，限时免费用于收集反馈 | ⭐⭐⭐⭐ |

### 📖 模型详细介绍

#### 1. **opencode/gpt-5-nano**
- **类型**：GPT-5 系列的精简版本
- **特点**：
  - ✅ 完全免费（输入/输出/缓存读取都免费）
  - ✅ 专为代码生成和工具调用优化
  - ✅ 轻量级，响应速度快
  - ✅ 适合日常开发任务
- **适用场景**：
  - 代码生成和补全
  - 代码解释和重构
  - 日常编程辅助
  - 学习和实验

#### 2. **opencode/big-pickle**
- **类型**：Stealth 模型（实验性）
- **特点**：
  - ✅ 完全免费（限时）
  - ✅ 用于收集用户反馈和改进模型
  - ✅ 可能在未来转为付费或调整
- **注意事项**：
  - ⚠️ 这是一个实验性模型
  - ⚠️ 数据可能用于模型改进（根据隐私政策）
  - ⚠️ 免费期限可能有限
- **适用场景**：
  - 尝试新功能
  - 提供反馈帮助改进模型
  - 实验性项目

### 📊 定价对比（来自 OpenCode Zen）

根据官方定价表，以下是免费模型与其他模型的对比：

| 模型 | 输入 (每100万tokens) | 输出 (每100万tokens) | 缓存读取 |
|------|---------------------|---------------------|---------|
| `opencode/gpt-5-nano` | **免费** | **免费** | **免费** |
| `opencode/big-pickle` | **免费** | **免费** | **免费** |
| `opencode/gpt-5.1-codex-mini` | $0.25 | $2.00 | $0.025 |
| `opencode/gpt-5.1` | $1.07 | $8.50 | $0.107 |
| `opencode/claude-haiku-3.5` | $0.80 | $4.00 | $0.08 |

---

## 🚀 使用示例

### 示例1：启动 OpenCode 并指定免费模型

```bash
cd /Users/h/practice/CV-main
opencode -m opencode/gpt-5-nano
```

### 示例2：在界面中切换模型

1. 启动 OpenCode：`opencode`
2. 按 `ctrl+x` 然后按 `m`（或 `<leader>m`）打开模型列表
3. 选择 `opencode/gpt-5-nano` 或 `opencode/big-pickle`

### 示例3：使用命令行直接运行

```bash
# 直接运行命令
opencode -m opencode/gpt-5-nano run "帮我写一个 Python 函数"

# 或进入交互模式
opencode -m opencode/gpt-5-nano
```

---

## 🔧 解决 "无可用渠道" 错误

如果你遇到类似错误：
```
分组 vip 下模型 gpt-5.2-chat-latest 无可用渠道（distributor）
```

**解决方案**：切换到免费模型

```bash
# 立即修复
opencode -m opencode/gpt-5-nano
```

---

## 📋 查看所有可用模型

```bash
# 查看所有模型
opencode models

# 只查看 OpenCode 免费模型
opencode models | grep "^opencode/"

# 当前可用的免费模型：
# - opencode/big-pickle
# - opencode/gpt-5-nano
```

### 🔍 模型列表说明

根据 OpenCode 官方文档，模型 ID 格式为 `provider_id/model_id`：

- **免费模型**：`opencode/` 开头的模型
  - `opencode/gpt-5-nano` - GPT-5 Nano（免费）
  - `opencode/big-pickle` - Big Pickle（免费，实验性）

- **其他模型**：需要相应的 API Key
  - `openai/` - 需要 OpenAI API Key
  - `anthropic/` - 需要 Anthropic API Key
  - 等等...

---

## 💡 使用技巧

### 1. 设置默认模型（可选）

如果你想每次启动都使用免费模型，可以创建一个别名：

```bash
# 添加到 ~/.zshrc
echo 'alias opencode-free="opencode -m opencode/gpt-5-nano"' >> ~/.zshrc
source ~/.zshrc

# 然后就可以直接使用
opencode-free
```

### 2. 在项目目录中使用

```bash
cd /Users/h/practice/CV-main
opencode -m opencode/gpt-5-nano
```

OpenCode 会自动识别项目结构，帮助你进行代码开发。

---

## ❓ 常见问题

### Q: opencode/gpt-5-nano 是什么模型？

A: 
- **GPT-5 Nano** 是 GPT-5 系列的精简版本
- 专为代码生成和工具调用优化
- 完全免费，无需 API Key
- 轻量级，响应速度快
- 适合日常开发任务

### Q: opencode/big-pickle 是什么模型？

A:
- **Big Pickle** 是一个 Stealth（实验性）模型
- 目前完全免费，用于收集用户反馈
- 数据可能用于模型改进（根据隐私政策）
- 免费期限可能有限

### Q: 免费模型和付费模型有什么区别？

A: 
- **免费模型**：
  - 无需 API Key
  - 功能完整，适合日常开发
  - 可能有速率限制
  - 完全免费使用
- **付费模型**：
  - 通常性能更强
  - 需要 API Key 和付费订阅
  - 更高的速率限制
  - 更多高级功能

### Q: 免费模型有限制吗？

A: 
- OpenCode 的免费模型通常有合理的速率限制
- 但对于个人开发完全够用
- 如果遇到限制，可以稍后再试

### Q: 如何知道当前使用的是哪个模型？

A: 
- 在 OpenCode 界面中，模型名称会显示在状态栏中
- 使用 `opencode models` 命令查看所有可用模型
- 使用 `opencode debug config` 查看当前配置

### Q: 需要注册 OpenCode Zen 才能使用免费模型吗？

A:
- 根据文档，OpenCode Zen 需要注册和 API Key
- 但 `opencode/gpt-5-nano` 和 `opencode/big-pickle` 可能在某些情况下可以直接使用
- 如果遇到问题，可以尝试注册 OpenCode Zen（免费注册）

---

## 🎉 开始使用

现在就试试吧！

```bash
cd /Users/h/practice/CV-main
opencode -m opencode/gpt-5-nano
```

**无需 API Key，开箱即用！** 🚀

---

## 📚 参考资源

- [OpenCode 官方文档](https://opencode.ai/docs/)
- [OpenCode Zen 文档](https://opencode.ai/docs/zen/)
- [OpenCode 模型文档](https://opencode.ai/docs/models/)
- [OpenCode GitHub](https://github.com/anomalyco/opencode)

---

*最后更新：2026-01-27*  
*参考：https://opencode.ai/docs/zen/*
