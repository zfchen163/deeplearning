# OpenCode 模型切换指南

## 🔴 问题描述

错误信息：`分组 vip 下模型 gpt-5.2-chat-latest 无可用渠道（distributor）`

**原因**：`gpt-5.2-chat-latest` 模型在当前配置下没有可用的访问渠道。

---

## 🆓 免费模型解决方案（无需 API Key）

### ⭐ 推荐：使用 OpenCode 内置免费模型

OpenCode 提供了**完全免费**的模型，**不需要任何 API Key**：

```bash
# 推荐：使用 gpt-5-nano（免费，已验证可用）
opencode -m opencode/gpt-5-nano

# 或使用 big-pickle（免费）
opencode -m opencode/big-pickle
```

**这些模型的特点**：
- ✅ **完全免费**，无需 API Key
- ✅ **无需注册**，开箱即用
- ✅ **已验证可用**，可以正常使用
- ✅ **适合代码生成**和日常开发

---

## ✅ 其他解决方案（需要 API Key）

### 方案1：切换到其他可用模型

#### 方法A：启动时指定模型
```bash
# 使用 gpt-5.1-chat-latest（需要 API Key）
opencode -m openai/gpt-5.1-chat-latest

# 或使用其他可用模型（需要 API Key）
opencode -m openai/gpt-5.1
opencode -m openai/gpt-5-pro
opencode -m openai/gpt-4o
opencode -m anthropic/claude-3-5-sonnet-latest
```

#### 方法B：在 OpenCode 界面中切换
1. 启动 OpenCode：`opencode`
2. 按 `<leader>m`（默认是 `ctrl+x` 然后按 `m`）打开模型列表
3. 选择其他可用的模型

---

### 方案2：检查并配置 API 密钥

如果必须使用 `gpt-5.2-chat-latest`，需要确保：

1. **检查环境变量**：
```bash
echo $OPENAI_API_KEY
```

2. **如果未设置，添加 API 密钥**：
```bash
# 临时设置（当前会话有效）
export OPENAI_API_KEY="your-api-key-here"

# 永久设置（添加到 ~/.zshrc）
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

3. **或使用 OpenCode 认证**：
```bash
opencode auth login
```

---

## 📋 可用的模型列表

### 🆓 免费模型（无需 API Key）

根据 [OpenCode Zen 官方文档](https://opencode.ai/docs/zen/)，以下模型完全免费：

```
✅ opencode/gpt-5-nano        # 推荐！GPT-5 Nano，完全免费
✅ opencode/big-pickle         # Big Pickle，实验性模型，限时免费
```

#### 模型详细介绍：

**opencode/gpt-5-nano**
- GPT-5 系列的精简版本
- 专为代码生成和工具调用优化
- 完全免费（输入/输出/缓存读取都免费）
- 轻量级，响应速度快
- 适合日常开发任务

**opencode/big-pickle**
- Stealth 模型（实验性）
- 完全免费（限时）
- 用于收集用户反馈和改进模型
- 数据可能用于模型改进（根据隐私政策）

### 💰 付费模型（需要 API Key）

```
✅ openai/gpt-5.1-chat-latest  # 需要 API Key
✅ openai/gpt-5.1
✅ openai/gpt-5.1-codex
✅ openai/gpt-5-pro
✅ openai/gpt-5
✅ openai/gpt-5-mini
✅ openai/gpt-5-nano
✅ openai/gpt-5-codex
❌ openai/gpt-5.2-chat-latest # 当前不可用
```

---

## 🚀 快速修复命令

### 对于没有 API Key 的用户（推荐）

**立即使用免费模型**：
```bash
# 使用完整路径（如果 PATH 未设置）
~/.opencode/bin/opencode -m opencode/gpt-5-nano

# 或如果 PATH 已设置
opencode -m opencode/gpt-5-nano
```

### 对于有 API Key 的用户

**切换到其他可用模型**：
```bash
# 使用完整路径（如果 PATH 未设置）
~/.opencode/bin/opencode -m openai/gpt-5.1-chat-latest

# 或如果 PATH 已设置
opencode -m openai/gpt-5.1-chat-latest
```

---

## 💡 其他可用模型推荐

### 🆓 免费模型（优先推荐）
- `opencode/gpt-5-nano` - **免费，无需 API Key，推荐使用**
- `opencode/big-pickle` - **免费，无需 API Key**

### 💰 付费模型（需要 API Key）

#### OpenAI 系列
- `openai/gpt-4o` - GPT-4 Optimized
- `openai/gpt-4o-mini` - 轻量级版本
- `openai/gpt-5-pro` - GPT-5 Pro 版本

#### Anthropic 系列
- `anthropic/claude-3-5-sonnet-latest` - Claude 3.5 Sonnet
- `anthropic/claude-3-7-sonnet-latest` - Claude 3.7 Sonnet
- `anthropic/claude-opus-4-5` - Claude Opus 4.5

---

## 🔍 查看所有可用模型

```bash
opencode models
```

---

## 📝 注意事项

1. **免费模型**：`opencode/gpt-5-nano` 和 `opencode/big-pickle` 完全免费，无需 API Key
2. **模型可用性**：某些付费模型可能需要特定的订阅或权限
3. **API 密钥**：使用付费模型时，确保你的 API 密钥有权限访问所选模型
4. **网络连接**：确保能正常访问模型服务

---

## 🆘 如果问题仍然存在

1. 检查 OpenCode 配置：
   ```bash
   opencode debug config
   ```

2. 查看认证状态：
   ```bash
   opencode auth list
   ```

3. 查看调试信息：
   ```bash
   opencode debug
   ```

---

*最后更新：2026-01-27*
