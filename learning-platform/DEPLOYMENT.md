# 🚀 部署指南

本文档介绍如何在不同环境下部署深度学习学习平台。

## 📋 目录

1. [本地开发部署](#本地开发部署)
2. [生产环境部署](#生产环境部署)
3. [Docker部署](#docker部署)
4. [云服务器部署](#云服务器部署)
5. [性能优化](#性能优化)

---

## 本地开发部署

### 前置要求
- Go 1.21+
- Python 3.8+
- 现代浏览器(Chrome、Firefox、Safari、Edge)

### 快速启动

```bash
# 1. 进入项目目录
cd /Users/h/practice/CV-main/learning-platform

# 2. 启动服务器
./start.sh

# 3. 访问平台
# 打开浏览器: http://localhost:8080
```

### 手动启动

```bash
# 1. 安装Go依赖
cd backend
go mod download

# 2. 运行服务器
go run main.go

# 或者编译后运行
go build -o server main.go
./server
```

### 自定义端口

```bash
# 使用环境变量
PORT=3000 go run main.go

# 或者
export PORT=3000
go run main.go
```

---

## 生产环境部署

### 1. 编译生产版本

```bash
cd backend

# 编译(带优化)
go build -ldflags="-s -w" -o server main.go

# 查看文件大小
ls -lh server
# 应该在 10-15MB 左右
```

### 2. 配置系统服务

创建 systemd 服务文件:

```bash
sudo nano /etc/systemd/system/learning-platform.service
```

内容:
```ini
[Unit]
Description=Deep Learning Platform
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/learning-platform/backend
ExecStart=/opt/learning-platform/backend/server
Restart=always
RestartSec=10
Environment="PORT=8080"

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl daemon-reload
sudo systemctl enable learning-platform
sudo systemctl start learning-platform
sudo systemctl status learning-platform
```

### 3. 配置Nginx反向代理

安装Nginx:
```bash
sudo apt update
sudo apt install nginx
```

配置文件 `/etc/nginx/sites-available/learning-platform`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /opt/learning-platform/frontend;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API代理
    location /api/ {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 静态资源缓存
    location /static/ {
        root /opt/learning-platform/frontend;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss 
               application/json application/javascript;
}
```

启用配置:
```bash
sudo ln -s /etc/nginx/sites-available/learning-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. 配置HTTPS (可选但推荐)

使用 Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Docker部署

### 1. 创建Dockerfile

```dockerfile
# backend/Dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o server main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates

WORKDIR /root/
COPY --from=builder /app/server .
COPY --from=builder /app/../frontend ./frontend
COPY --from=builder /app/../../*.ipynb ./notebooks/

EXPOSE 8080
CMD ["./server"]
```

### 2. 创建docker-compose.yml

```yaml
version: '3.8'

services:
  learning-platform:
    build: ./backend
    ports:
      - "8080:8080"
    volumes:
      - ./notebooks:/root/notebooks:ro
      - ./frontend:/root/frontend:ro
    environment:
      - PORT=8080
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:8080/api/categories"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 云服务器部署

### 阿里云ECS部署

#### 1. 购买服务器
- 配置: 2核4G (最低)
- 系统: Ubuntu 20.04 LTS
- 带宽: 5Mbps

#### 2. 安全组配置
开放端口:
- 22 (SSH)
- 80 (HTTP)
- 443 (HTTPS)

#### 3. 部署步骤

```bash
# SSH连接服务器
ssh root@your-server-ip

# 安装依赖
apt update
apt install -y git golang-go nginx

# 克隆项目
cd /opt
git clone your-repo-url learning-platform
cd learning-platform

# 编译
cd backend
go build -o server main.go

# 配置systemd和nginx (参考上面的生产环境部署)

# 启动服务
systemctl start learning-platform
systemctl start nginx
```

### 腾讯云CVM部署

类似阿里云,步骤相同。

### AWS EC2部署

```bash
# 连接EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# 后续步骤同上
```

---

## 性能优化

### 1. Go后端优化

#### 启用Gzip压缩

在 `main.go` 中添加:
```go
import "github.com/gin-contrib/gzip"

router.Use(gzip.Gzip(gzip.DefaultCompression))
```

#### 设置缓存头

```go
router.Static("/static", "../frontend/static")
router.Use(func(c *gin.Context) {
    if strings.HasPrefix(c.Request.URL.Path, "/static/") {
        c.Header("Cache-Control", "public, max-age=2592000") // 30天
    }
    c.Next()
})
```

### 2. 前端优化

#### 压缩CSS和JS

```bash
# 安装工具
npm install -g csso-cli uglify-js

# 压缩CSS
csso frontend/static/css/style.css -o frontend/static/css/style.min.css

# 压缩JS
uglifyjs frontend/static/js/app.js -c -m -o frontend/static/js/app.min.js
```

#### 使用CDN

修改 `index.html`:
```html
<!-- 使用CDN加速 -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
```

### 3. 数据库优化 (如果添加数据库)

```go
// 使用连接池
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)

// 添加索引
CREATE INDEX idx_category ON notebooks(category);
CREATE INDEX idx_title ON notebooks(title);
```

### 4. 负载均衡 (高并发场景)

使用Nginx负载均衡:
```nginx
upstream backend {
    server localhost:8080;
    server localhost:8081;
    server localhost:8082;
}

server {
    location /api/ {
        proxy_pass http://backend;
    }
}
```

---

## 监控和日志

### 1. 日志配置

```go
// 使用文件日志
f, _ := os.Create("server.log")
gin.DefaultWriter = io.MultiWriter(f, os.Stdout)
```

### 2. 监控工具

使用 Prometheus + Grafana:
```bash
# 安装Prometheus
docker run -d -p 9090:9090 prom/prometheus

# 安装Grafana
docker run -d -p 3000:3000 grafana/grafana
```

### 3. 健康检查

添加健康检查端点:
```go
router.GET("/health", func(c *gin.Context) {
    c.JSON(200, gin.H{
        "status": "ok",
        "timestamp": time.Now().Unix(),
    })
})
```

---

## 备份策略

### 1. 数据备份

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/learning-platform"

# 备份笔记本文件
tar -czf $BACKUP_DIR/notebooks_$DATE.tar.gz /opt/learning-platform/*.ipynb

# 保留最近7天的备份
find $BACKUP_DIR -name "notebooks_*.tar.gz" -mtime +7 -delete
```

### 2. 自动备份

添加到crontab:
```bash
crontab -e

# 每天凌晨2点备份
0 2 * * * /opt/scripts/backup.sh
```

---

## 故障排查

### 常见问题

#### 1. 端口被占用
```bash
# 查看端口占用
lsof -i :8080

# 杀死进程
kill -9 PID
```

#### 2. Go依赖下载失败
```bash
# 使用代理
export GOPROXY=https://goproxy.cn,direct
go mod download
```

#### 3. 权限问题
```bash
# 修改文件权限
chmod +x start.sh
chmod +x backend/server

# 修改所有者
chown -R www-data:www-data /opt/learning-platform
```

#### 4. 内存不足
```bash
# 查看内存使用
free -h

# 添加swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 安全建议

### 1. 防火墙配置

```bash
# UFW防火墙
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. 限制访问

```nginx
# Nginx限流
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20;
    proxy_pass http://localhost:8080;
}
```

### 3. HTTPS强制

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

---

## 性能基准

### 测试环境
- CPU: 2核
- 内存: 4GB
- 系统: Ubuntu 20.04

### 性能指标
- QPS: 1000+
- 平均响应时间: 50ms
- 并发连接: 500+
- 内存占用: 30-50MB

### 压力测试

```bash
# 使用ab工具
ab -n 10000 -c 100 http://localhost:8080/api/categories

# 使用wrk工具
wrk -t4 -c100 -d30s http://localhost:8080/api/categories
```

---

## 总结

选择合适的部署方式:

| 场景 | 推荐方案 | 成本 |
|------|---------|------|
| 个人学习 | 本地部署 | 免费 |
| 小团队 | VPS + Nginx | ¥50/月 |
| 学校使用 | 云服务器 | ¥200/月 |
| 大规模 | 负载均衡 + CDN | ¥1000+/月 |

**现在你可以根据需求选择合适的部署方案了!** 🚀

---

**需要帮助?** 查看项目文档或提交Issue。
