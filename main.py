from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DataStorage:
    def __init__(self):
        self.raw_data = []
        self.ahp_framework = {}
        self.weights = {}
        self.prompts = []
        self.concepts = []
        self.evaluations = []
        
    def add_raw_data(self, data):
        self.raw_data.append(data)
        return len(self.raw_data) - 1
    
    def set_ahp_framework(self, framework):
        self.ahp_framework = framework
    
    def set_weights(self, weights):
        self.weights = weights
    
    def add_prompt(self, prompt):
        self.prompts.append(prompt)
        return len(self.prompts) - 1
    
    def add_concept(self, concept):
        self.concepts.append(concept)
        return len(self.concepts) - 1

storage = DataStorage()

# 文档中的默认数据
DEFAULT_CRITERIA = ['功能性', '隐私性', '人性化', '环境适应性']
DEFAULT_SUB_CRITERIA = {
    '功能性': ['模块化', '智能化', '集成化', '可扩展性'],
    '隐私性': ['数据安全', '访问控制', '信息保密', '物理隔离'],
    '人性化': ['可达性', '易用性', '舒适性', '情感化'],
    '环境适应性': ['气候适应', '地形适应', '能源适应', '空间适应']
}

# 文档中的默认权重
DEFAULT_WEIGHTS = {
    '功能性': 0.47236,
    '隐私性': 0.08233,
    '人性化': 0.22789,
    '环境适应性': 0.21743
}

DEFAULT_SUB_WEIGHTS = {
    '功能性': [0.4866, 0.0848, 0.1213, 0.3073],
    '隐私性': [0.0996, 0.3820, 0.4320, 0.0864],
    '人性化': [0.4573, 0.1052, 0.1377, 0.2998],
    '环境适应性': [0.4444, 0.1597, 0.1597, 0.2361]
}

def calculate_ahp(matrix):
    """按照文档公式计算AHP矩阵的权重和一致性"""
    try:
        if not matrix or len(matrix) == 0:
            return {
                'weights': [],
                'max_eigenvalue': 0,
                'CI': 1,
                'CR': 1,
                'consistency': False,
                'error': '矩阵为空'
            }
        
        matrix = np.array(matrix, dtype=float)
        
        if matrix.ndim == 1:
            n = int(np.sqrt(len(matrix)))
            if n * n != len(matrix):
                raise ValueError("矩阵长度必须是平方数")
            matrix = matrix.reshape((n, n))
        
        n = len(matrix)
        
        # 按照文档公式：几何平均法计算权重
        geometric_means = []
        for i in range(n):
            row_product = 1.0
            for j in range(n):
                row_product *= matrix[i][j]
            geometric_means.append(row_product ** (1/n))
        
        total_sum = sum(geometric_means)
        if total_sum == 0:
            return {
                'weights': [],
                'max_eigenvalue': 0,
                'CI': 1,
                'CR': 1,
                'consistency': False,
                'error': '权重计算错误：总和为0'
            }
        
        weights = [gm / total_sum for gm in geometric_means]
        weights = np.array(weights)
        
        # 计算最大特征值 λ_max
        weighted_sum = np.dot(matrix, weights)
        max_eigenvalue = 0
        for i in range(n):
            if weights[i] != 0:
                max_eigenvalue += weighted_sum[i] / weights[i]
        max_eigenvalue = max_eigenvalue / n
        
        # 一致性检验 CI = (λ_max - n) / (n - 1)
        CI = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        
        # RI值表（文档表2）
        RI_dict = {
            1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.12,
            6: 1.26, 7: 1.36, 8: 1.41, 9: 1.45, 10: 1.49
        }
        
        RI = RI_dict.get(n, 1.49)
        CR = CI / RI if RI != 0 else 0
        
        # 文档中的计算公式
        formulas = {
            'geometric_mean': "w_i = (∏_{j=1}^n a_{ij})^{1/n} / Σ_{k=1}^n (∏_{j=1}^n a_{kj})^{1/n}",
            'lambda_max': "λ_max = (1/n) * Σ_{i=1}^n (Σ_{j=1}^n a_{ij}w_j) / w_i",
            'CI': "CI = (λ_max - n) / (n - 1)",
            'CR': "CR = CI / RI"
        }
        
        return {
            'weights': [float(w) for w in weights],
            'max_eigenvalue': float(max_eigenvalue),
            'CI': float(CI),
            'CR': float(CR),
            'consistency': bool(CR < 0.1),
            'formulas': formulas
        }
        
    except Exception as e:
        print(f"AHP计算错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'weights': [],
            'max_eigenvalue': 0,
            'CI': 1,
            'CR': 1,
            'consistency': False,
            'error': str(e)
        }

def calculate_gra_topsis(matrix, weights, impacts):
    """按照文档公式计算GRA-TOPSIS"""
    try:
        m = len(matrix)  # 方案数
        n = len(matrix[0])  # 指标数
        
        # 转换为numpy数组
        X = np.array(matrix, dtype=float)
        
        # 步骤1：无量纲化处理（公式3）- 平方和归一化
        # c_{ij} = X_{ij} / sqrt(Σ_{i=1}^m X_{ij}^2)
        norm_matrix = np.zeros_like(X)
        for j in range(n):
            col_sum_sq = np.sum(X[:, j] ** 2)
            if col_sum_sq > 0:
                norm_matrix[:, j] = X[:, j] / np.sqrt(col_sum_sq)
        
        # 步骤2：加权规范化决策矩阵 D = C * W
        weighted_matrix = norm_matrix * weights
        
        # 步骤3：参考序列（取最大值）
        reference_seq = weighted_matrix.max(axis=0)
        
        # 步骤4：计算灰色关联系数（公式5）
        rho = 0.5  # 分辨系数
        min_diff = np.min(np.abs(weighted_matrix - reference_seq))
        max_diff = np.max(np.abs(weighted_matrix - reference_seq))
        
        grey_rel_matrix = np.zeros_like(weighted_matrix)
        for i in range(m):
            for j in range(n):
                diff = abs(weighted_matrix[i, j] - reference_seq[j])
                if min_diff == max_diff:
                    grey_rel_matrix[i, j] = 1
                else:
                    grey_rel_matrix[i, j] = (min_diff + rho * max_diff) / (diff + rho * max_diff)
        
        # 步骤5：TOPSIS计算
        # 正理想解和负理想解
        ideal_best = grey_rel_matrix.max(axis=0)
        ideal_worst = grey_rel_matrix.min(axis=0)
        
        # 计算距离（公式7）
        dist_best = np.sqrt(np.sum((grey_rel_matrix - ideal_best) ** 2, axis=1))
        dist_worst = np.sqrt(np.sum((grey_rel_matrix - ideal_worst) ** 2, axis=1))
        
        # 相对贴近度（公式8）
        closeness = dist_worst / (dist_best + dist_worst + 1e-10)
        
        # 转换为百分制
        scores = (closeness * 100).tolist()
        
        # 计算排名
        sorted_indices = np.argsort(-closeness)
        rank = sorted_indices.tolist()
        
        # 文档中的计算公式
        formulas = {
            'normalization': "c_{ij} = X_{ij} / √(Σ_{i=1}^m X_{ij}^2)",
            'weighted': "d_{ij} = c_{ij} × w_j",
            'grey_relation': "ζ_i(k) = (min_i min_k Δ_i(k) + ρ·max_i max_k Δ_i(k)) / (Δ_i(k) + ρ·max_i max_k Δ_i(k))",
            'ideal_solution': "ζ_0^+ = max ζ_i(k), ζ_0^- = min ζ_i(k)",
            'distance': "l_i^+ = √(Σ_{k=1}^m (ζ_i(k) - ζ_0^+(k))^2), l_i^- = √(Σ_{k=1}^n (ζ_i(k) - ζ_0^-(k))^2)",
            'closeness': "U_i^+ = l_i^- / (l_i^+ + l_i^-)"
        }
        
        # 详细的计算步骤
        calculation_steps = {
            'normalized_matrix': norm_matrix.tolist(),
            'weighted_matrix': weighted_matrix.tolist(),
            'reference_sequence': reference_seq.tolist(),
            'grey_relation_matrix': grey_rel_matrix.tolist(),
            'ideal_best': ideal_best.tolist(),
            'ideal_worst': ideal_worst.tolist(),
            'distances_best': dist_best.tolist(),
            'distances_worst': dist_worst.tolist()
        }
        
        return {
            'closeness': closeness.tolist(),
            'rank': rank,
            'scores': scores,
            'formulas': formulas,
            'calculation_steps': calculation_steps
        }
    except Exception as e:
        print(f"GRA-TOPSIS计算错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'closeness': [],
            'rank': [],
            'scores': [],
            'formulas': {},
            'calculation_steps': {}
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demand-analysis', methods=['GET', 'POST'])
def demand_analysis():
    if request.method == 'POST':
        data_type = request.form.get('data_type')
        
        # 使用文档中的社区医疗车数据
        extracted_data = {
            'metadata': [
                {
                    'title': '社区医疗车设计研究综述',
                    'authors': '张华, 李明, 王芳',
                    'institution': '湖北工业大学工业设计工程学院',
                    'references': 45
                },
                {
                    'title': '移动医疗设施可达性研究',
                    'authors': '陈伟, 刘洋',
                    'institution': '清华大学设计学院',
                    'references': 32
                },
                {
                    'title': '智能医疗车的人机交互设计',
                    'authors': '赵雪, 孙强',
                    'institution': '浙江大学工业设计系',
                    'references': 28
                }
            ],
            'pain_points': [
                '偏远地区医疗服务覆盖不足',
                '老年人使用数字医疗设备困难',
                '医疗车空间利用率低',
                '设备维护成本高'
            ],
            'emotional_needs': [
                '安全感 - 担心医疗设备不可靠',
                '信任感 - 对远程医疗的疑虑',
                '便利性 - 希望一站式服务',
                '亲切感 - 希望医疗环境温馨'
            ],
            'keywords': ['功能性', '隐私性', '人性化', '环境适应性', '模块化', '智能化', '可达性', '安全性']
        }
        
        session['extracted_data'] = extracted_data
        return jsonify({'success': True, 'data': extracted_data})
    
    return render_template('demand_analysis.html')

@app.route('/ahp-evaluation', methods=['GET', 'POST'])
def ahp_evaluation():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            try:
                data = request.get_json()
                
                if 'matrix' not in data or 'criteria' not in data:
                    return jsonify({
                        'error': '缺少必要参数',
                        'consistency': False
                    })
                
                matrix = data['matrix']
                criteria = data['criteria']
                
                result = calculate_ahp(matrix)
                
                # 保存权重
                if result.get('weights') and len(result['weights']) > 0:
                    weights = {}
                    for i, criterion in enumerate(criteria):
                        if i < len(result['weights']):
                            weights[criterion] = result['weights'][i]
                    
                    storage.set_weights(weights)
                    session['ahp_weights'] = weights
                    session['ahp_criteria'] = criteria
                
                return jsonify(result)
                
            except Exception as e:
                print(f"AHP路由错误: {e}")
                return jsonify({
                    'error': f'服务器错误: {str(e)}',
                    'consistency': False
                })
        else:
            try:
                # 使用文档中的默认框架
                framework = {
                    'goal': '社区医疗车设计优化',
                    'criteria': DEFAULT_CRITERIA,
                    'sub_criteria': DEFAULT_SUB_CRITERIA
                }
                
                storage.set_ahp_framework(framework)
                session['ahp_framework'] = framework
                session['ahp_criteria'] = framework['criteria']
                
                # 设置默认权重
                storage.set_weights(DEFAULT_WEIGHTS)
                session['ahp_weights'] = DEFAULT_WEIGHTS
                
                return jsonify({'success': True, 'framework': framework})
                
            except Exception as e:
                print(f"框架提交错误: {e}")
                return jsonify({'success': False, 'error': str(e)})
    
    extracted_data = session.get('extracted_data', {})
    criteria = session.get('ahp_criteria', DEFAULT_CRITERIA)
    weights = session.get('ahp_weights', DEFAULT_WEIGHTS)
    
    # 构建默认矩阵（文档表4）
    default_matrix = [
        [1, 2, 3, 5],
        [1/2, 1, 1/2, 4],
        [1/3, 2, 1, 2],
        [1/5, 1/4, 1/2, 1]
    ]
    
    return render_template('ahp_evaluation.html', 
                         extracted_data=extracted_data, 
                         criteria=criteria,
                         default_weights=weights,
                         default_matrix=default_matrix)

@app.route('/prompt-generation', methods=['GET', 'POST'])
def prompt_generation():
    if request.method == 'POST':
        prompt_data = {
            'constraints': request.form.get('constraints'),
            'style': request.form.get('style'),
            'scene': request.form.get('scene'),
            'technical_terms': request.form.get('technical_terms'),
            'ai_tool': request.form.get('ai_tool', 'Midjourney'),
            'image_quality': request.form.get('image_quality', 'high'),
            'num_images': int(request.form.get('num_images', 4)),
            'style_consistency': float(request.form.get('style_consistency', 0.8))
        }
        
        # 使用文档中表6的提示词
        final_prompt = """模块化医疗车在社区区域扩展了三层车厢，全息健康数据屏幕显示实时分析结果，车顶上的六角形太阳能电池板为设备充电，无人机从可伸缩舱发射，全地形轮胎在潮湿的沥青路面上留下胎痕，亚光灰色车身与住宅楼融为一体，隐藏式车底防护装置，琥珀色引导灯照亮轮椅坡道，老人使用带有大图标的语音辅助触摸屏，抗菌半透明窗帘划分区域，动画云排队系统 --ar 3:2 --s 750 --style raw --no sci-fi,neon,clean_tires"""
        
        prompt_id = storage.add_prompt({
            'data': prompt_data,
            'prompt': final_prompt
        })
        
        return jsonify({
            'success': True,
            'prompt_id': prompt_id,
            'final_prompt': final_prompt
        })
    
    weights = storage.weights or session.get('ahp_weights', DEFAULT_WEIGHTS)
    return render_template('prompt_generation.html', weights=weights)

@app.route('/concept-screening', methods=['GET', 'POST'])
def concept_screening():
    if request.method == 'POST':
        if 'action' in request.form:
            if request.form['action'] == 'generate':
                # 使用文档中的方案描述
                concepts = []
                concept_descriptions = [
                    "模块化医疗车扩展三层车厢，车顶太阳能板，全息数据屏",
                    "全地形医疗车，隐藏式防护，琥珀色引导灯，语音交互",
                    "智能医疗车，无人机舱，抗菌隔帘，云排队系统"
                ]
                
                for i in range(3):
                    concepts.append({
                        'id': i,
                        'title': f'方案 P{i+1}',
                        'description': concept_descriptions[i],
                        'innovation_score': [85, 78, 92][i],
                        'feasibility_score': [82, 85, 79][i],
                        'weight_coverage': [88, 76, 91][i],
                        'image_url': f'/static/concept_{i+1}.jpg'
                    })
                
                session['concepts'] = concepts
                return jsonify({'success': True, 'concepts': concepts})
            
            elif request.form['action'] == 'select':
                selected = request.form.getlist('selected[]')
                concepts = session.get('concepts', [])
                selected_concepts = [c for c in concepts if str(c['id']) in selected]
                session['selected_concepts'] = selected_concepts[:3]
                
                return jsonify({
                    'success': True,
                    'selected': [c['title'] for c in selected_concepts[:3]]
                })
    
    concepts = session.get('concepts', [])
    return render_template('concept_screening.html', concepts=concepts)

@app.route('/design-evaluation', methods=['GET', 'POST'])
def design_evaluation():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = request.get_json()
            if 'scores' in data:
                scores = np.array(data['scores'])
                weights = np.array(data['weights'])
                impacts = data['impacts']
                
                # 使用文档中的GRA-TOPSIS公式
                result = calculate_gra_topsis(scores, weights, impacts)
                return jsonify(result)
        else:
            scores = []
            for i in range(3):
                scheme_scores = []
                for j in range(16):
                    score_key = f'scores[{i}][{j}]'
                    if score_key in request.form:
                        scheme_scores.append(float(request.form[score_key]))
                scores.append(scheme_scores)
            
            session['evaluation_scores'] = scores
            return jsonify({'success': True, 'scores': scores})
    
    concepts = session.get('selected_concepts', [])
    
    # 如果没有选中的概念，使用默认方案
    if not concepts:
        concepts = [
            {'title': '方案 P1', 'description': '模块化医疗车扩展三层车厢'},
            {'title': '方案 P2', 'description': '全地形医疗车，隐藏式防护'},
            {'title': '方案 P3', 'description': '智能医疗车，无人机舱'}
        ]
    
    framework = session.get('ahp_framework', {})
    
    return render_template('design_evaluation.html', 
                         concepts=concepts, 
                         framework=framework)

@app.route('/api/save-feedback', methods=['POST'])
def save_feedback():
    data = request.get_json()
    return jsonify({'success': True})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, port=5000, host='127.0.0.1')