# 深度学习在线学习平台

一个专为高中生设计的深度学习入门课程平台,使用 Go 语言构建后端,原生 JavaScript 构建前端。

## ✨ 特性

- 🎓 **157个精心优化的课程** - 从基础到进阶,系统学习深度学习
- 🎮 **趣味学习** - 用生活中的例子和类比,让抽象概念变得具体
- 💻 **动手实践** - 每个概念都配有可运行的代码示例
- 📊 **分类清晰** - 9大主题分类,循序渐进
- 🔍 **快速搜索** - 快速找到你想学的内容
- 📱 **响应式设计** - 支持电脑、平板、手机访问

## 🚀 快速开始

### 前置要求

- Go 1.21 或更高版本
- Python 3.8+ (用于笔记本优化)

### 安装步骤

1. **克隆或进入项目目录**
```bash
cd /Users/h/practice/CV-main/learning-platform
```

2. **安装Go依赖**
```bash
cd backend
go mod download
```

3. **优化笔记本(首次运行)**
```bash
cd ..
python3 optimize_all_notebooks.py
```

4. **启动服务器**
```bash
cd backend
go run main.go
```

5. **访问平台**
打开浏览器访问: http://localhost:8080

## 📁 项目结构

```
learning-platform/
├── backend/              # Go后端
│   ├── main.go          # 主程序
│   └── go.mod           # Go依赖
├── frontend/            # 前端
│   ├── index.html       # 主页面
│   └── static/          # 静态资源
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
└── README.md
```

## 🎯 课程分类

1. **基础入门** - 环境搭建、Python基础
2. **数据处理** - 数据加载、预处理、增强
3. **神经网络基础** - 感知机、激活函数、损失函数
4. **卷积神经网络** - CNN、LeNet、ResNet等经典网络
5. **循环神经网络** - RNN、LSTM、GRU
6. **注意力机制** - Attention、Transformer、BERT
7. **计算机视觉** - 目标检测、图像分割、风格迁移
8. **实战项目** - Kaggle竞赛、真实项目
9. **高级主题** - 分布式训练、模型优化、大模型

## 🔧 API接口

### 获取分类列表
```
GET /api/categories
```

### 获取笔记本列表
```
GET /api/notebooks?category=卷积神经网络
```

### 获取笔记本内容
```
GET /api/notebook/:filename
```

### 搜索笔记本
```
GET /api/search?q=卷积
```

## 💡 使用技巧

1. **从基础开始** - 按照分类顺序学习,不要跳过基础课程
2. **动手实践** - 每个代码示例都要运行一遍
3. **修改参数** - 试着改变参数,观察结果变化
4. **做笔记** - 记录你的理解和疑问
5. **坚持学习** - 每天30分钟,3个月后你会看到显著进步

## 🛠️ 技术栈

- **后端**: Go + Gin框架
- **前端**: 原生JavaScript + HTML5 + CSS3
- **代码高亮**: Highlight.js
- **Markdown渲染**: Marked.js
- **笔记本格式**: Jupyter Notebook (.ipynb)

## 📝 开发说明

### 添加新课程

1. 将 `.ipynb` 文件放到项目根目录
2. 运行优化脚本:
```bash
python3 optimize_all_notebooks.py
```
3. 重启服务器

### 自定义分类

编辑 `backend/main.go` 中的 `categories` 和 `categoryKeywords` 变量。

### 修改端口

```bash
PORT=3000 go run main.go
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

## 👨‍💻 作者

深度学习学习平台团队

---

**记住:** 学习是一个循序渐进的过程,不要着急,慢慢来! 💪
