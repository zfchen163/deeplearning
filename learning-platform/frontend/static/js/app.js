// API基础URL
const API_BASE = '/api';

// 状态管理
const state = {
    categories: [],
    currentNotebook: null,
    searchResults: []
};

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    loadCategories();
    setupEventListeners();
});

// 设置事件监听器
function setupEventListeners() {
    // 搜索按钮
    document.getElementById('searchBtn').addEventListener('click', handleSearch);
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });

    // 返回按钮
    document.getElementById('backBtn').addEventListener('click', showWelcomePage);
    document.getElementById('closeSearchBtn').addEventListener('click', showWelcomePage);

    // 折叠全部按钮
    document.getElementById('collapseAllBtn').addEventListener('click', toggleAllCategories);
}

// 加载分类列表
async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE}/categories`);
        const data = await response.json();
        state.categories = data.categories;
        renderCategories(data.categories);
        updateTotalNotebooks(data.categories);
    } catch (error) {
        console.error('加载分类失败:', error);
        showError('加载课程列表失败,请刷新页面重试');
    }
}

// 渲染分类列表
function renderCategories(categories) {
    const container = document.getElementById('categoriesList');
    
    if (!categories || categories.length === 0) {
        container.innerHTML = '<div class="loading">暂无课程</div>';
        return;
    }

    container.innerHTML = categories.map(category => `
        <div class="category-item">
            <div class="category-header" onclick="toggleCategory('${category.name}')">
                <div>
                    <div class="category-name">${category.name}</div>
                    <div class="category-intro">${category.intro}</div>
                </div>
                <span class="category-count">${category.notebooks.length}</span>
            </div>
            <div class="notebooks-list" id="notebooks-${category.name}" style="display: none;">
                ${category.notebooks.map(notebook => `
                    <div class="notebook-item" onclick="loadNotebook('${notebook.filename}', '${notebook.title}', '${category.name}')">
                        <span class="notebook-order">${notebook.order}.</span>
                        <span class="notebook-title">${notebook.title}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

// 切换分类展开/折叠
function toggleCategory(categoryName) {
    const notebooksList = document.getElementById(`notebooks-${categoryName}`);
    const header = notebooksList.previousElementSibling;
    
    if (notebooksList.style.display === 'none') {
        notebooksList.style.display = 'block';
        header.classList.remove('collapsed');
    } else {
        notebooksList.style.display = 'none';
        header.classList.add('collapsed');
    }
}

// 折叠/展开所有分类
function toggleAllCategories() {
    const allLists = document.querySelectorAll('.notebooks-list');
    const allHeaders = document.querySelectorAll('.category-header');
    const btn = document.getElementById('collapseAllBtn');
    
    const isAnyExpanded = Array.from(allLists).some(list => list.style.display === 'block');
    
    allLists.forEach((list, index) => {
        if (isAnyExpanded) {
            list.style.display = 'none';
            allHeaders[index].classList.add('collapsed');
        } else {
            list.style.display = 'block';
            allHeaders[index].classList.remove('collapsed');
        }
    });
    
    btn.textContent = isAnyExpanded ? '展开全部' : '折叠全部';
}

// 加载笔记本内容
async function loadNotebook(filename, title, category) {
    try {
        // 显示加载状态
        showNotebookViewer();
        document.getElementById('notebookTitle').textContent = title;
        document.getElementById('notebookCategory').textContent = category;
        document.getElementById('notebookContent').innerHTML = '<div class="loading">加载中</div>';

        // 高亮当前笔记本
        document.querySelectorAll('.notebook-item').forEach(item => {
            item.classList.remove('active');
        });
        event.currentTarget.classList.add('active');

        // 获取笔记本内容
        const response = await fetch(`${API_BASE}/notebook/${filename}`);
        const notebook = await response.json();
        
        state.currentNotebook = { filename, title, category, content: notebook };
        renderNotebook(notebook);
        
        // 滚动到顶部
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
        console.error('加载笔记本失败:', error);
        document.getElementById('notebookContent').innerHTML = 
            '<div class="error">加载失败,请重试</div>';
    }
}

// 渲染笔记本内容
function renderNotebook(notebook) {
    const container = document.getElementById('notebookContent');
    
    if (!notebook.cells || notebook.cells.length === 0) {
        container.innerHTML = '<div class="loading">笔记本为空</div>';
        return;
    }

    container.innerHTML = notebook.cells.map((cell, index) => {
        if (cell.cell_type === 'markdown') {
            return renderMarkdownCell(cell, index);
        } else if (cell.cell_type === 'code') {
            return renderCodeCell(cell, index);
        }
        return '';
    }).join('');

    // 应用代码高亮
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

// 渲染Markdown单元格
function renderMarkdownCell(cell, index) {
    let source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    
    // 处理图片附件
    if (cell.attachments) {
        for (const [filename, attachment] of Object.entries(cell.attachments)) {
            // 获取图片数据
            for (const [mimeType, data] of Object.entries(attachment)) {
                if (mimeType.startsWith('image/')) {
                    // 创建data URL
                    const dataUrl = `data:${mimeType};base64,${data}`;
                    // 替换Markdown中的附件引用
                    const attachmentPattern = new RegExp(`!\\[([^\\]]*)\\]\\(attachment:${filename}\\)`, 'g');
                    source = source.replace(attachmentPattern, `![$1](${dataUrl})`);
                }
            }
        }
    }
    
    const html = marked.parse(source);
    
    return `
        <div class="cell cell-markdown" data-index="${index}">
            ${html}
        </div>
    `;
}

// 渲染代码单元格
function renderCodeCell(cell, index) {
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    
    return `
        <div class="cell cell-code" data-index="${index}">
            <div class="code-header">
                <span class="code-label">Python</span>
                <button class="copy-btn" onclick="copyCode(${index})">📋 复制代码</button>
            </div>
            <div class="code-content">
                <pre><code class="language-python">${escapeHtml(source)}</code></pre>
            </div>
        </div>
    `;
}

// 复制代码
function copyCode(index) {
    const cell = state.currentNotebook.content.cells[index];
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    
    navigator.clipboard.writeText(source).then(() => {
        const btn = event.currentTarget;
        const originalText = btn.textContent;
        btn.textContent = '✅ 已复制';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        console.error('复制失败:', err);
        alert('复制失败,请手动复制');
    });
}

// 搜索笔记本
async function handleSearch() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        alert('请输入搜索关键词');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        state.searchResults = data.results || [];
        showSearchResults(data.results, query);
    } catch (error) {
        console.error('搜索失败:', error);
        showError('搜索失败,请重试');
    }
}

// 显示搜索结果
function showSearchResults(results, query) {
    hideAllPages();
    document.getElementById('searchResults').style.display = 'block';
    
    const container = document.getElementById('searchResultsList');
    
    if (!results || results.length === 0) {
        container.innerHTML = `
            <div class="loading">
                没有找到包含 "${query}" 的课程<br>
                试试其他关键词吧!
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div style="margin-bottom: 20px; color: var(--text-secondary);">
            找到 <strong>${results.length}</strong> 个相关课程
        </div>
        ${results.map(notebook => `
            <div class="search-result-item" onclick="loadNotebook('${notebook.filename}', '${notebook.title}', '${notebook.category}')">
                <div class="search-result-title">${highlightText(notebook.title, query)}</div>
                <span class="search-result-category">${notebook.category}</span>
            </div>
        `).join('')}
    `;
}

// 高亮搜索关键词
function highlightText(text, query) {
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
}

// 显示欢迎页面
function showWelcomePage() {
    hideAllPages();
    document.getElementById('welcomePage').style.display = 'block';
    
    // 清除高亮
    document.querySelectorAll('.notebook-item').forEach(item => {
        item.classList.remove('active');
    });
}

// 显示笔记本查看器
function showNotebookViewer() {
    hideAllPages();
    document.getElementById('notebookViewer').style.display = 'block';
}

// 隐藏所有页面
function hideAllPages() {
    document.getElementById('welcomePage').style.display = 'none';
    document.getElementById('notebookViewer').style.display = 'none';
    document.getElementById('searchResults').style.display = 'none';
}

// 更新总课程数
function updateTotalNotebooks(categories) {
    const total = categories.reduce((sum, cat) => sum + cat.notebooks.length, 0);
    document.getElementById('totalNotebooks').textContent = total;
}

// 显示错误信息
function showError(message) {
    alert(message);
}

// HTML转义
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// 配置marked选项
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: true,
    mangle: false
});
