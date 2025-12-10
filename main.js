

// AHP矩阵处理
class AHPProcessor {
    constructor() {
        this.matrix = [];
        this.criteria = [];
    }
    
    initMatrix(size) {
        this.matrix = Array(size).fill().map(() => Array(size).fill(1));
        for (let i = 0; i < size; i++) {
            this.matrix[i][i] = 1;
        }
        return this.matrix;
    }
    
    updateValue(row, col, value) {
        if (row >= 0 && row < this.matrix.length && col >= 0 && col < this.matrix.length) {
            this.matrix[row][col] = parseFloat(value);
            this.matrix[col][row] = 1 / parseFloat(value);
        }
    }
    
    validateMatrix() {
        for (let i = 0; i < this.matrix.length; i++) {
            for (let j = 0; j < this.matrix.length; j++) {
                if (isNaN(this.matrix[i][j]) || this.matrix[i][j] <= 0) {
                    return false;
                }
            }
        }
        return true;
    }
}

// TOPSIS计算
class TOPSISCalculator {
    static calculate(scores, weights, impacts) {
        // 标准化
        const normScores = scores.map(row => 
            row.map((val, idx) => 
                val / Math.sqrt(scores.reduce((sum, r) => sum + Math.pow(r[idx], 2), 0))
            )
        );
        
        // 加权
        const weightedScores = normScores.map(row =>
            row.map((val, idx) => val * weights[idx])
        );
        
        // 理想解
        const idealBest = weightedScores[0].map((_, idx) =>
            impacts[idx] === '+' 
                ? Math.max(...weightedScores.map(row => row[idx]))
                : Math.min(...weightedScores.map(row => row[idx]))
        );
        
        const idealWorst = weightedScores[0].map((_, idx) =>
            impacts[idx] === '+'
                ? Math.min(...weightedScores.map(row => row[idx]))
                : Math.max(...weightedScores.map(row => row[idx]))
        );
        
        // 计算距离
        const distances = weightedScores.map(row => {
            const distBest = Math.sqrt(
                row.reduce((sum, val, idx) => sum + Math.pow(val - idealBest[idx], 2), 0)
            );
            const distWorst = Math.sqrt(
                row.reduce((sum, val, idx) => sum + Math.pow(val - idealWorst[idx], 2), 0)
            );
            return distWorst / (distBest + distWorst);
        });
        
        // 排序
        const ranked = distances.map((score, idx) => ({ idx, score }))
            .sort((a, b) => b.score - a.score)
            .map(item => item.idx);
        
        return {
            closeness: distances,
            rank: ranked,
            scores: distances.map(score => score * 100)
        };
    }
}

// 概念筛选管理器
class ConceptManager {
    constructor() {
        this.selected = new Set();
    }
    
    toggleConcept(id) {
        if (this.selected.has(id)) {
            this.selected.delete(id);
        } else {
            this.selected.add(id);
        }
        this.updateUI();
    }
    
    updateUI() {
        document.querySelectorAll('.concept-card').forEach(card => {
            const id = card.dataset.id;
            if (this.selected.has(id)) {
                card.classList.add('selected');
                card.querySelector('.concept-checkbox').checked = true;
            } else {
                card.classList.remove('selected');
                card.querySelector('.concept-checkbox').checked = false;
            }
        });
    }
    
    getSelected() {
        return Array.from(this.selected);
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
    
    // 全局事件监听
    initGlobalEvents();
});

function initGlobalEvents() {
    // 表单提交处理
    document.addEventListener('submit', function(e) {
        const form = e.target;
        if (form.classList.contains('ajax-form')) {
            e.preventDefault();
            submitAjaxForm(form);
        }
    });
    
    // 动态元素事件委托
    document.addEventListener('click', function(e) {
        // 概念选择
        if (e.target.classList.contains('concept-checkbox')) {
            const card = e.target.closest('.concept-card');
            card.classList.toggle('selected');
        }
        
        // 复制按钮
        if (e.target.classList.contains('btn-copy')) {
            const text = e.target.dataset.copy;
            navigator.clipboard.writeText(text).then(() => {
                Utils.showAlert('已复制到剪贴板', 'success');
            });
        }
    });
}

async function submitAjaxForm(form) {
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 处理中...';
    
    try {
        const response = await fetch(form.action, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            Utils.showAlert('操作成功', 'success');
            if (result.redirect) {
                setTimeout(() => {
                    window.location.href = result.redirect;
                }, 1000);
            }
        } else {
            Utils.showAlert(result.message || '操作失败', 'danger');
        }
    } catch (error) {
        Utils.showAlert('网络错误，请重试', 'danger');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}