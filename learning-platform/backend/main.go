package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Notebook 笔记本结构
type Notebook struct {
	Filename string `json:"filename"`
	Title    string `json:"title"`
	Category string `json:"category"`
	Order    int    `json:"order"`
	Path     string `json:"path"`
}

// NotebookContent 笔记本内容
type NotebookContent struct {
	Cells []Cell `json:"cells"`
}

// Cell 单元格
type Cell struct {
	CellType       string                 `json:"cell_type"`
	Source         interface{}            `json:"source"`
	Outputs        []interface{}          `json:"outputs,omitempty"`
	ExecutionCount interface{}            `json:"execution_count,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	Attachments    map[string]interface{} `json:"attachments,omitempty"`
}

// Category 分类
type Category struct {
	Name     string     `json:"name"`
	Intro    string     `json:"intro"`
	Notebooks []Notebook `json:"notebooks"`
}

var (
	notebooksDir = "../../" // 笔记本目录
	categories   = map[string]string{
		"基础入门":   "🚀 从零开始,搭建你的深度学习环境",
		"数据处理":   "📊 数据是AI的燃料,学会处理数据是第一步",
		"神经网络基础": "🧠 理解神经网络的基本组件",
		"卷积神经网络": "🖼️ 让计算机\"看懂\"图片的秘密武器",
		"循环神经网络": "🔄 处理时间序列和文本的神经网络",
		"注意力机制":  "👀 让AI学会\"关注重点\"",
		"计算机视觉":  "👁️ 图像识别、物体检测等视觉任务",
		"实战项目":   "💪 真实项目实战,检验学习成果",
		"高级主题":   "🚀 进阶技术和前沿应用",
	}
	categoryKeywords = map[string][]string{
		"基础入门":   {"配置", "安装", "Python", "Pytorch", "START"},
		"数据处理":   {"数据", "Dataloader", "Transforms", "预处理", "增广"},
		"神经网络基础": {"感知机", "线性", "激活", "损失", "优化器", "反向传播"},
		"卷积神经网络": {"卷积", "池化", "LeNet", "AlexNet", "VGG", "ResNet", "GoogLeNet"},
		"循环神经网络": {"RNN", "LSTM", "GRU", "序列", "循环"},
		"注意力机制":  {"注意力", "Transformer", "BERT", "seq2seq"},
		"计算机视觉":  {"检测", "分割", "识别", "风格迁移", "目标检测"},
		"实战项目":   {"Kaggle", "竞赛", "实战", "项目"},
		"高级主题":   {"分布式", "GPU", "TPU", "微调", "RAG", "大模型"},
	}
)

func main() {
	// 设置Gin为发布模式
	gin.SetMode(gin.ReleaseMode)

	router := gin.Default()

	// 配置CORS
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"*"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept"}
	router.Use(cors.New(config))

	// 静态文件服务
	router.Static("/static", "../frontend/static")
	router.StaticFile("/", "../frontend/index.html")

	// API路由
	api := router.Group("/api")
	{
		api.GET("/categories", getCategories)
		api.GET("/notebooks", getNotebooks)
		api.GET("/notebook/:filename", getNotebookContent)
		api.GET("/search", searchNotebooks)
	}

	// 启动服务器
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("\n🚀 学习平台启动成功!\n")
	fmt.Printf("📚 访问地址: http://localhost:%s\n\n", port)

	if err := router.Run(":" + port); err != nil {
		log.Fatal("服务器启动失败:", err)
	}
}

// 获取分类列表
func getCategories(c *gin.Context) {
	notebooks := scanNotebooks()
	categorized := categorizeNotebooks(notebooks)

	var result []Category
	for name, intro := range categories {
		if nbs, ok := categorized[name]; ok {
			result = append(result, Category{
				Name:      name,
				Intro:     intro,
				Notebooks: nbs,
			})
		}
	}

	// 按预定义顺序排序
	categoryOrder := []string{"基础入门", "数据处理", "神经网络基础", "卷积神经网络", "循环神经网络", "注意力机制", "计算机视觉", "实战项目", "高级主题"}
	sort.Slice(result, func(i, j int) bool {
		iIdx := indexOf(categoryOrder, result[i].Name)
		jIdx := indexOf(categoryOrder, result[j].Name)
		if iIdx == -1 {
			iIdx = 999
		}
		if jIdx == -1 {
			jIdx = 999
		}
		return iIdx < jIdx
	})

	c.JSON(http.StatusOK, gin.H{
		"categories": result,
	})
}

// 获取所有笔记本
func getNotebooks(c *gin.Context) {
	category := c.Query("category")
	notebooks := scanNotebooks()

	if category != "" {
		categorized := categorizeNotebooks(notebooks)
		if nbs, ok := categorized[category]; ok {
			c.JSON(http.StatusOK, gin.H{
				"notebooks": nbs,
			})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"notebooks": notebooks,
	})
}

// 获取笔记本内容
func getNotebookContent(c *gin.Context) {
	filename := c.Param("filename")

	// 安全检查:防止路径遍历攻击
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无效的文件名",
		})
		return
	}

	filePath := filepath.Join(notebooksDir, filename)

	// 检查文件是否存在
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "笔记本不存在",
		})
		return
	}

	// 读取文件内容
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "读取文件失败",
		})
		return
	}

	// 解析JSON
	var notebook NotebookContent
	if err := json.Unmarshal(data, &notebook); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "解析笔记本失败",
		})
		return
	}

	c.JSON(http.StatusOK, notebook)
}

// 搜索笔记本
func searchNotebooks(c *gin.Context) {
	query := strings.ToLower(c.Query("q"))
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "搜索关键词不能为空",
		})
		return
	}

	notebooks := scanNotebooks()
	var results []Notebook

	for _, nb := range notebooks {
		if strings.Contains(strings.ToLower(nb.Title), query) ||
			strings.Contains(strings.ToLower(nb.Filename), query) {
			results = append(results, nb)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"results": results,
		"count":   len(results),
	})
}

// 扫描笔记本目录
func scanNotebooks() []Notebook {
	var notebooks []Notebook

	files, err := ioutil.ReadDir(notebooksDir)
	if err != nil {
		log.Printf("读取目录失败: %v", err)
		return notebooks
	}

	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".ipynb") &&
			!strings.HasSuffix(file.Name(), "_backup.ipynb") {

			title := extractTitle(file.Name())
			category := categorizeNotebook(file.Name())
			order := extractOrder(file.Name())

			notebooks = append(notebooks, Notebook{
				Filename: file.Name(),
				Title:    title,
				Category: category,
				Order:    order,
				Path:     filepath.Join(notebooksDir, file.Name()),
			})
		}
	}

	// 按顺序排序
	sort.Slice(notebooks, func(i, j int) bool {
		return notebooks[i].Order < notebooks[j].Order
	})

	return notebooks
}

// 提取标题
func extractTitle(filename string) string {
	name := strings.TrimSuffix(filename, ".ipynb")
	re := regexp.MustCompile(`^\d+_`)
	name = re.ReplaceAllString(name, "")
	return name
}

// 提取顺序
func extractOrder(filename string) int {
	re := regexp.MustCompile(`^(\d+)_`)
	matches := re.FindStringSubmatch(filename)
	if len(matches) > 1 {
		var order int
		fmt.Sscanf(matches[1], "%d", &order)
		return order
	}
	return 999
}

// 分类笔记本
func categorizeNotebook(filename string) string {
	filenameLower := strings.ToLower(filename)

	for category, keywords := range categoryKeywords {
		for _, keyword := range keywords {
			if strings.Contains(filenameLower, strings.ToLower(keyword)) {
				return category
			}
		}
	}

	return "其他"
}

// 按分类组织笔记本
func categorizeNotebooks(notebooks []Notebook) map[string][]Notebook {
	result := make(map[string][]Notebook)

	for _, nb := range notebooks {
		result[nb.Category] = append(result[nb.Category], nb)
	}

	return result
}

// 辅助函数:查找索引
func indexOf(slice []string, item string) int {
	for i, v := range slice {
		if v == item {
			return i
		}
	}
	return -1
}
