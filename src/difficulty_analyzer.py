"""
难度分析模块
负责分析生成题目的难度等级
"""
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer


class QuestionDifficultyAnalyzer:
    """题目难度分析器"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        初始化难度分析器
        
        Args:
            model_name: 用于语义分析的模型名称
        """
        # 初始化语言模型
        self.model = SentenceTransformer(model_name)
        
        # 定义难度等级
        self.difficulty_levels = {
            0: "🟢 简单(1星)",
            1: "🟡 基础(2星)", 
            2: "🟠 中等(3星)",
            3: "🔴 困难(4星)",
            4: "⭐ 挑战(5星)"
        }
        
        # 通信专业术语列表
        self.technical_terms = [
            'MIMO',  # 多输入多输出
            'CSI',   # 信道状态信息
            'SNR',   # 信噪比
            '信源',  # 信息源
            '信道',  # 通信信道
            '编码',  # 编码
            '解码',  # 解码
            '语义',  # 语义
            'OFDM',  # 正交频分复用
            'QAM',   # 正交幅度调制
            '波束成形',  # Beamforming
            'NOMA',  # 非正交多址接入
            'FEC',   # 前向纠错
            '毫米波',  # mmWave
            'URLLC', # 超高可靠低时延通信
            '网络切片',  # Network Slicing
            '极化码',  # Polar Codes
            '干扰对齐',  # Interference Alignment
            '边缘计算',  # Edge Computing
            'C-RAN',  # 云无线接入网
            '多径衰落',  # Multipath Fading
            '频谱共享',  # Spectrum Sharing
        ]
    
    def extract_options(self, question_text: str) -> List[str]:
        """
        提取题目中的选项信息
        
        Args:
            question_text: 题目文本
            
        Returns:
            选项文本列表
        """
        options = []
        lines = question_text.split('\n')
        
        for line in lines:
            # 查找形如 "- A." "- B." 等的选项行
            stripped_line = line.strip()
            if stripped_line.startswith('- ') and '. ' in stripped_line:
                option = stripped_line[2:]  # 去掉"- "前缀
                if len(option) >= 2 and option[0].isalpha() and option[1] == '.':
                    option_text = option[2:].strip()
                    if option_text:
                        options.append(option_text)
        
        return options
    
    def calculate_option_similarity(self, options: List[str]) -> float:
        """
        计算选项之间的相似度
        
        Args:
            options: 选项列表
            
        Returns:
            平均相似度
        """
        if not options or len(options) < 2:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        # 计算所有选项对之间的相似度
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                similarity = SequenceMatcher(None, options[i], options[j]).ratio()
                total_similarity += similarity
                comparisons += 1
        
        # 返回平均相似度
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def analyze_complexity_features(self, question_text: str) -> Tuple[float, float]:
        """
        分析题目复杂度特征
        
        Args:
            question_text: 题目文本
            
        Returns:
            (长度分数, 术语分数) 的元组
        """
        # 计算题目长度分数 (归一化)
        length_score = min(len(question_text) / 1000, 1.0)
        
        # 检测专业术语数量
        term_count = sum(1 for term in self.technical_terms if term in question_text)
        term_score = min(term_count / 10, 1.0)  # 归一化术语分数
        
        return length_score, term_score
    
    def get_difficulty_level(self, similarity_score: float, length_score: float, term_score: float) -> str:
        """
        确定难度等级(1-5星)
        
        Args:
            similarity_score: 相似度得分 (0-1，相似度越低难度越高)
            length_score: 长度得分 (0-1)
            term_score: 术语得分 (0-1)
        
        Returns:
            难度等级描述
        """
        # 综合评分 (相似度越低难度越高)
        total_score = (1 - similarity_score) * 0.5 + length_score * 0.2 + term_score * 0.3
        
        # 设定明确的分数区间对应1-5星
        if total_score < 0.2:
            return self.difficulty_levels[0]  # 1星 - 非常简单
        elif total_score < 0.4:
            return self.difficulty_levels[1]  # 2星 - 简单
        elif total_score < 0.6:
            return self.difficulty_levels[2]  # 3星 - 中等
        elif total_score < 0.8:
            return self.difficulty_levels[3]  # 4星 - 困难
        else:
            return self.difficulty_levels[4]  # 5星 - 非常困难
    
    def analyze_question(self, question_text: str) -> Dict[str, any]:
        """
        分析单个题目的难度
        
        Args:
            question_text: 题目文本
            
        Returns:
            包含分析结果的字典
        """
        # 提取选项
        options = self.extract_options(question_text)
        
        # 计算各项指标
        similarity_score = self.calculate_option_similarity(options)
        length_score, term_score = self.analyze_complexity_features(question_text)
        
        # 确定难度等级
        difficulty = self.get_difficulty_level(similarity_score, length_score, term_score)
        
        analysis = {
            "选项数量": len(options),
            "选项相似度": f"{similarity_score:.3f}",
            "题目长度得分": f"{length_score:.3f}",
            "专业术语得分": f"{term_score:.3f}",
            "难度等级": difficulty,
            "综合得分": f"{(1 - similarity_score) * 0.5 + length_score * 0.2 + term_score * 0.3:.3f}"
        }
        
        return analysis
    
    def format_question(self, question_text: str) -> str:
        """
        格式化题目文本，包含选项
        
        Args:
            question_text: 原始题目文本
            
        Returns:
            格式化后的题目文本
        """
        lines = question_text.strip().split('\n')
        question = lines[0].strip()
        options = [line.strip() for line in lines if line.strip().startswith('- ')]
        return question + '\n' + '\n'.join(options)
    
    def analyze_file_and_update(self, markdown_file: str, output_file: str = None) -> None:
        """
        分析文件中的所有题目并更新难度等级
        
        Args:
            markdown_file: 输入的markdown文件路径
            output_file: 输出文件路径，如果为None则覆盖原文件
        """
        # 读取原始文件
        content = Path(markdown_file).read_text(encoding='utf-8')
        
        # 提取所有题目块
        sections = re.split(r'\n(\d+\. \[选择题\])', content)
        new_content = sections[0]  # 保留文件开头
        
        # 处理每个题目
        i = 1
        while i < len(sections):
            if i + 1 < len(sections):
                # 获取题目编号和题目内容
                question_header = sections[i]
                question_block = sections[i + 1]
                
                # 格式化题目文本
                question_text = self.format_question(question_header + question_block)
                
                # 分析题目难度
                analysis = self.analyze_question(question_text)
                print(f"题目分析 {analysis}")
                
                # 在题目编号后添加难度等级
                updated_header = question_header.replace(
                    "[选择题]", 
                    f"[选择题] {analysis['难度等级']}"
                )
                
                new_content += updated_header + question_block
                i += 2
            else:
                new_content += sections[i]
                i += 1
        
        # 写入文件
        if output_file is None:
            output_file = markdown_file
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"难度分析完成，结果已保存到 {output_file}")
    
    def generate_difficulty_report(self, markdown_file: str, output_file: str = None) -> Dict[str, any]:
        """
        生成难度分析报告
        
        Args:
            markdown_file: markdown文件路径
            output_file: 报告输出文件路径
            
        Returns:
            包含统计信息的字典
        """
        # 读取文件并提取题目
        content = Path(markdown_file).read_text(encoding='utf-8')
        question_blocks = re.findall(r'\d+\. \[选择题\](.*?)(?=\n\d+\. \[选择题\]|\Z)', content, re.DOTALL)
        
        # 分析每道题目
        analyses = []
        difficulty_distribution = {level: 0 for level in self.difficulty_levels.values()}
        
        for i, block in enumerate(question_blocks, 1):
            question_text = self.format_question(block)
            analysis = self.analyze_question(question_text)
            analyses.append({
                "question_id": i,
                **analysis
            })
            
            # 统计难度分布
            difficulty_level = analysis["难度等级"]
            if difficulty_level in difficulty_distribution:
                difficulty_distribution[difficulty_level] += 1
        
        # 生成统计信息
        stats = {
            "总题目数": len(analyses),
            "难度分布": difficulty_distribution,
            "平均综合得分": np.mean([float(a["综合得分"]) for a in analyses]),
            "题目详情": analyses
        }
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# 题目难度分析报告\n\n")
                
                f.write("## 总体统计\n")
                f.write(f"- 总题目数: {stats['总题目数']}\n")
                f.write(f"- 平均综合得分: {stats['平均综合得分']:.3f}\n\n")
                
                f.write("## 难度分布\n")
                for level, count in stats['难度分布'].items():
                    percentage = count / stats['总题目数'] * 100
                    f.write(f"- {level}: {count} ({percentage:.1f}%)\n")
                
                f.write("\n## 详细分析\n\n")
                for analysis in analyses:
                    f.write(f"### 题目 {analysis['question_id']}\n")
                    for key, value in analysis.items():
                        if key != "question_id":
                            f.write(f"- {key}: {value}\n")
                    f.write("\n")
        
        print(f"难度分析报告已生成")
        return stats


def get_analyzer_config() -> dict:
    """获取分析器配置"""
    return {
        "model_name": "BAAI/bge-m3",
        "weights": {
            "similarity": 0.5,
            "length": 0.2,
            "technical_terms": 0.3
        },
        "difficulty_thresholds": {
            "very_easy": 0.2,
            "easy": 0.4,
            "medium": 0.6,
            "hard": 0.8
        }
    }