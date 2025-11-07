"""
质量评估模块
负责评估生成题目的质量
"""
import re
import jieba
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SurveyEvaluator:
    """问卷质量评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.rouge = Rouge()
        self.tfidf = TfidfVectorizer()
    
    def extract_questions_and_sources(self, markdown_file: str) -> Tuple[List[str], List[str]]:
        """
        从markdown文件中提取所有选择题和出处
        
        Args:
            markdown_file: markdown文件路径
            
        Returns:
            题目列表和来源文本列表的元组
        """
        content = Path(markdown_file).read_text(encoding='utf-8')
        
        # 提取所有选择题块
        question_blocks = re.findall(
            r'\d+\. \[选择题\](.*?)(?=\n\d+\. \[选择题\]|\Z)', 
            content, 
            re.DOTALL
        )
        
        all_questions = []
        all_source_texts = []
        
        for i, block in enumerate(question_blocks, 1):
            # 提取选择题内容和对应的出处
            source_match = re.search(r'出处：{(.*?)}', block)
            if source_match:
                full_question = f"{i}. [选择题]{block}"
                source_text = source_match.group(1).strip()
                all_questions.append(full_question)
                all_source_texts.append(source_text)
        
        return all_questions, all_source_texts
    
    def _extract_question_text(self, question_data: str) -> str:
        """
        提取问题相关文本
        
        Args:
            question_data: 完整的问题数据
            
        Returns:
            问题文本
        """
        try:
            if "出处：" in question_data:
                question_text = question_data.split("出处：")[1].strip("{}")
            else:
                # 如果没有出处，提取题目部分
                lines = question_data.split('\n')
                question_text = '\n'.join([
                    line for line in lines 
                    if line.strip() and not line.startswith('- ') and '正确答案' not in line
                ])
            return question_text
        except Exception:
            return question_data
    
    def evaluate_question(self, question_data: str, source_text: str) -> Dict[str, float]:
        """
        评估单个题目的质量
        
        Args:
            question_data: 题目数据
            source_text: 来源文本
            
        Returns:
            包含各项评分的字典
        """
        # 提取问题相关文本
        question_text = self._extract_question_text(question_data)
        
        # 1. ROUGE-L 评分 (占比 30%)
        try:
            rouge_scores = self.rouge.get_scores(question_text, source_text)[0]
            rouge_l_score = rouge_scores["rouge-l"]["f"] * 0.3
        except Exception:
            rouge_l_score = 0.0
        
        # 2. 关键词覆盖率评分 (占比 30%)
        try:
            source_keywords = set(jieba.cut(source_text.lower()))
            question_keywords = set(jieba.cut(question_text.lower()))
            if len(source_keywords) > 0:
                keyword_coverage = len(question_keywords.intersection(source_keywords)) / len(source_keywords) * 0.3
            else:
                keyword_coverage = 0.0
        except Exception:
            keyword_coverage = 0.0
        
        # 3. 语义相似度评分 (占比 30%)
        try:
            texts = [question_text, source_text]
            if texts[0] and texts[1]:  # 确保文本不为空
                tfidf_matrix = self.tfidf.fit_transform(texts)
                semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 0.3
            else:
                semantic_similarity = 0.0
        except Exception:
            semantic_similarity = 0.0
        
        # 4. 题目格式完整性评分 (占比 10%)
        format_score = self._evaluate_format(question_data) * 0.1
        
        # 计算总分
        total_score = rouge_l_score + keyword_coverage + semantic_similarity + format_score
        
        return {
            "total_score": total_score,
            "rouge_l": rouge_l_score,
            "keyword_coverage": keyword_coverage,
            "semantic_similarity": semantic_similarity,
            "format_score": format_score
        }
    
    def _evaluate_format(self, question_data: str) -> float:
        """
        评估题目格式的完整性
        
        Args:
            question_data: 题目数据
            
        Returns:
            格式分数
        """
        score = 1.0
        required_elements = [
            "[选择题]",
            "- A.",
            "- B.",
            "- C.",
            "- D.",
            "正确答案：",
            "出处："
        ]
        
        for element in required_elements:
            if element not in question_data:
                score -= 1.0 / len(required_elements)
        
        return max(0, score)
    
    def validate_survey(self, markdown_file: str) -> List[Dict]:
        """
        评估整个问卷的质量
        
        Args:
            markdown_file: markdown文件路径
            
        Returns:
            包含每道题评估结果的列表
        """
        questions, source_texts = self.extract_questions_and_sources(markdown_file)
        results = []
        
        print(f"开始评估问卷，共 {len(questions)} 道题目...")
        
        # 遍历所有选择题进行评分
        for i, (question, source) in enumerate(zip(questions, source_texts)):
            try:
                score = self.evaluate_question(question, source)
                results.append({
                    "question_id": i + 1,
                    "scores": score,
                    "quality": "合格" if score["total_score"] >= 0.6 else "需要改进"
                })
            except Exception as e:
                print(f"评估题目 {i+1} 时出错: {e}")
                results.append({
                    "question_id": i + 1,
                    "scores": {"total_score": 0.0, "rouge_l": 0.0, "keyword_coverage": 0.0, "semantic_similarity": 0.0, "format_score": 0.0},
                    "quality": "评估失败"
                })
        
        print("问卷评估完成！")
        return results
    
    def print_evaluation_results(self, results: List[Dict]) -> None:
        """
        打印评估结果
        
        Args:
            results: 评估结果列表
        """
        total_questions = len(results)
        qualified_count = sum(1 for result in results if result["quality"] == "合格")
        avg_score = np.mean([result["scores"]["total_score"] for result in results])
        
        print(f"\n=== 问卷质量评估报告 ===")
        print(f"总题目数: {total_questions}")
        print(f"合格题目数: {qualified_count}")
        print(f"合格率: {qualified_count/total_questions*100:.1f}%")
        print(f"平均分数: {avg_score:.2f}")
        print("="*30)
        
        for result in results:
            print(f"\n评估题目 {result['question_id']}:")
            print(f"总分: {result['scores']['total_score']:.2f}")
            print(f"质量: {result['quality']}")
            print("详细分数:")
            print(f"- ROUGE-L: {result['scores']['rouge_l']:.2f}")
            print(f"- 关键词覆盖: {result['scores']['keyword_coverage']:.2f}")
            print(f"- 语义相似度: {result['scores']['semantic_similarity']:.2f}")
            print(f"- 格式完整性: {result['scores']['format_score']:.2f}")
    
    def save_evaluation_report(self, results: List[Dict], output_file: str) -> None:
        """
        保存评估报告到文件
        
        Args:
            results: 评估结果列表
            output_file: 输出文件路径
        """
        total_questions = len(results)
        qualified_count = sum(1 for result in results if result["quality"] == "合格")
        avg_score = np.mean([result["scores"]["total_score"] for result in results])
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 问卷质量评估报告\n\n")
            f.write(f"## 总体统计\n")
            f.write(f"- 总题目数: {total_questions}\n")
            f.write(f"- 合格题目数: {qualified_count}\n")
            f.write(f"- 合格率: {qualified_count/total_questions*100:.1f}%\n")
            f.write(f"- 平均分数: {avg_score:.2f}\n\n")
            
            f.write("## 详细评估结果\n\n")
            
            for result in results:
                f.write(f"### 题目 {result['question_id']}\n")
                f.write(f"- **总分**: {result['scores']['total_score']:.2f}\n")
                f.write(f"- **质量**: {result['quality']}\n")
                f.write("- **详细分数**:\n")
                f.write(f"  - ROUGE-L: {result['scores']['rouge_l']:.2f}\n")
                f.write(f"  - 关键词覆盖: {result['scores']['keyword_coverage']:.2f}\n")
                f.write(f"  - 语义相似度: {result['scores']['semantic_similarity']:.2f}\n")
                f.write(f"  - 格式完整性: {result['scores']['format_score']:.2f}\n\n")
        
        print(f"评估报告已保存至 {output_file}")


def get_evaluator_config() -> dict:
    """获取评估器配置"""
    return {
        "passing_score": 0.6,
        "weights": {
            "rouge_l": 0.3,
            "keyword_coverage": 0.3,
            "semantic_similarity": 0.3,
            "format": 0.1
        }
    }