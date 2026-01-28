#!/bin/bash

echo "🚀 启动深度学习学习平台..."
echo ""

# 检查Go是否安装
if ! command -v go &> /dev/null; then
    echo "❌ 错误: 未找到Go语言环境"
    echo "请先安装Go: https://golang.org/dl/"
    exit 1
fi

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    echo "请先安装Python3"
    exit 1
fi

# 进入后端目录
cd backend

# 检查是否需要下载依赖
if [ ! -d "vendor" ] && [ ! -f "go.sum" ]; then
    echo "📦 正在下载Go依赖..."
    go mod download
    echo "✅ 依赖下载完成"
    echo ""
fi

# 启动服务器
echo "🌟 启动服务器..."
echo ""
go run main.go
