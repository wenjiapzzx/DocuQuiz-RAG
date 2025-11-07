"""
主程序入口
整合所有模块，提供完整的文档自动出题功能
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 导入自定义模块
from config import initialize_environment, load_embedding_model, get_default_config
from llm import setup_deepseek_llm
from document_processor import DocumentProcessor, HybridRetriever
from quiz_generator import QuizGenerator
from evaluator import SurveyEvaluator
from difficulty_analyzer import QuestionDifficultyAnalyzer


class DocuQuizSystem:
    """文档自动出题系统主类"""
    
    def __init__(
        self,
        config: Optional[dict] = None,
        llama_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None
    ):
        """
        初始化文档出题系统
        
        Args:
            config: 配置字典
            llama_api_key: LlamaParse API密钥
            deepseek_api_key: DeepSeek API密钥
        """
        # 设置配置
        self.config = config or get_default_config()
        
        # 初始化环境
        print("正在初始化环境...")
        initialize_environment(
            hf_home=self.config.get("hf_home"),
            hf_endpoint=self.config.get("hf_endpoint")
        )
        
        # 加载嵌入模型
        print("正在加载嵌入模型...")
        load_embedding_model(
            model_name=self.config.get("model_name"),
            cache_folder=self.config.get("hf_home")
        )
        
        # 设置LLM
        print("正在设置LLM...")
        setup_deepseek_llm(api_key=deepseek_api_key)
        
        # 初始化文档处理器
        self.doc_processor = DocumentProcessor(
            llama_api_key=llama_api_key,
            persist_dir=self.config.get("vector_store_dir"),
            cache_dir=self.config.get("document_cache_dir")
        )
        
        # 初始化其他组件
        self.quiz_generator = None
        self.evaluator = SurveyEvaluator()
        self.difficulty_analyzer = QuestionDifficultyAnalyzer()
        
        print("系统初始化完成！")
    
    def process_documents(self, pdf_paths: List[str], force_rebuild: bool = False) -> None:
        """
        处理文档并构建索引
        
        Args:
            pdf_paths: PDF文件路径列表
            force_rebuild: 是否强制重建索引
        """
        # 检查是否需要重建索引
        if not force_rebuild:
            try:
                print("尝试加载已存在的索引...")
                self.doc_processor.load_index()
                self.doc_processor.nodes = self.doc_processor.load_cached_documents()
                print("索引加载成功！")
                return
            except FileNotFoundError:
                print("未找到已存在的索引，开始构建...")
        
        # 解析文档
        print("开始解析文档...")
        documents = self.doc_processor.parse_documents(pdf_paths)
        
        # 构建索引
        print("开始构建索引...")
        self.doc_processor.build_index(documents)
    
    def setup_quiz_generator(self, vector_weight: float = 0.6) -> None:
        """
        设置题目生成器
        
        Args:
            vector_weight: 向量检索权重
        """
        if not self.doc_processor.index:
            raise ValueError("请先处理文档并构建索引")
        
        # 创建检索器
        vector_retriever, bm25_retriever = self.doc_processor.create_retrievers()
        
        # 创建混合检索器
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            vector_weight=vector_weight
        )
        
        # 创建题目生成器
        self.quiz_generator = QuizGenerator(retriever=hybrid_retriever)
        
        print("题目生成器设置完成！")
    
    def generate_quiz(
        self,
        query: str,
        num_questions: int = 6,
        output_file: str = "survey.md",
        title: str = "通信文档问卷"
    ) -> str:
        """
        生成题目
        
        Args:
            query: 查询关键词
            num_questions: 生成题目数量
            output_file: 输出文件路径
            title: 问卷标题
            
        Returns:
            输出文件路径
        """
        if not self.quiz_generator:
            raise ValueError("请先设置题目生成器")
        
        print(f"正在为查询 '{query}' 生成 {num_questions} 道题目...")
        
        # 生成题目
        questions = self.quiz_generator.generate_questions(
            query=query,
            num_questions=num_questions
        )
        
        # 保存题目
        self.quiz_generator.save_quiz_to_file(
            questions=questions,
            output_file=output_file,
            title=title
        )
        
        return output_file
    
    def evaluate_quiz(self, quiz_file: str) -> List[dict]:
        """
        评估题目质量
        
        Args:
            quiz_file: 题目文件路径
            
        Returns:
            评估结果列表
        """
        print("正在评估题目质量...")
        
        results = self.evaluator.validate_survey(quiz_file)
        self.evaluator.print_evaluation_results(results)
        
        # 保存评估报告
        report_file = quiz_file.replace('.md', '_evaluation_report.md')
        self.evaluator.save_evaluation_report(results, report_file)
        
        return results
    
    def analyze_difficulty(self, quiz_file: str, update_file: bool = True) -> dict:
        """
        分析题目难度
        
        Args:
            quiz_file: 题目文件路径
            update_file: 是否更新原文件
            
        Returns:
            分析统计信息
        """
        print("正在分析题目难度...")
        
        if update_file:
            # 更新原文件
            self.difficulty_analyzer.analyze_file_and_update(quiz_file)
        
        # 生成难度报告
        report_file = quiz_file.replace('.md', '_difficulty_report.md')
        stats = self.difficulty_analyzer.generate_difficulty_report(quiz_file, report_file)
        
        # 打印统计信息
        print(f"\n=== 难度分析统计 ===")
        print(f"总题目数: {stats['总题目数']}")
        print(f"平均综合得分: {stats['平均综合得分']:.3f}")
        print("难度分布:")
        for level, count in stats['难度分布'].items():
            percentage = count / stats['总题目数'] * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        return stats
    
    def run_full_pipeline(
        self,
        pdf_paths: List[str],
        queries: List[str],
        output_dir: str = "./output",
        force_rebuild: bool = False,
        evaluate: bool = True,
        analyze_difficulty: bool = True
    ) -> List[str]:
        """
        运行完整流程
        
        Args:
            pdf_paths: PDF文件路径列表
            queries: 查询关键词列表
            output_dir: 输出目录
            force_rebuild: 是否强制重建索引
            evaluate: 是否进行质量评估
            analyze_difficulty: 是否进行难度分析
            
        Returns:
            生成的文件路径列表
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        # 1. 处理文档
        self.process_documents(pdf_paths, force_rebuild)
        
        # 2. 设置题目生成器
        self.setup_quiz_generator()
        
        # 3. 生成题目
        for query in queries:
            output_file = os.path.join(output_dir, f"quiz_{query.replace(' ', '_')}.md")
            
            try:
                self.generate_quiz(
                    query=query,
                    num_questions=6,
                    output_file=output_file,
                    title=f"通信文档问卷 - {query}"
                )
                generated_files.append(output_file)
                
                # 4. 评估题目质量
                if evaluate:
                    self.evaluate_quiz(output_file)
                
                # 5. 分析题目难度
                if analyze_difficulty:
                    self.analyze_difficulty(output_file)
                
            except Exception as e:
                print(f"处理查询 '{query}' 时出错: {e}")
        
        print(f"\n=== 流程完成 ===")
        print(f"生成的文件: {generated_files}")
        return generated_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DocuQuiz-RAG 文档自动出题系统")
    
    parser.add_argument(
        "--pdf_paths", 
        nargs="+", 
        default=["./data/语义通信白皮书.pdf"],
        help="PDF文件路径列表"
    )
    
    parser.add_argument(
        "--queries",
        nargs="+", 
        default=["多模态语义通信", "信道编码", "语义理解"],
        help="查询关键词列表"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./output", 
        help="输出目录"
    )
    
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="强制重建索引"
    )
    
    parser.add_argument(
        "--no_evaluate",
        action="store_true",
        help="跳过质量评估"
    )
    
    parser.add_argument(
        "--no_difficulty",
        action="store_true", 
        help="跳过难度分析"
    )
    
    parser.add_argument(
        "--llama_api_key",
        help="LlamaParse API密钥"
    )
    
    parser.add_argument(
        "--deepseek_api_key",
        help="DeepSeek API密钥"
    )
    
    args = parser.parse_args()
    
    try:
        # 初始化系统
        system = DocuQuizSystem(
            llama_api_key=args.llama_api_key,
            deepseek_api_key=args.deepseek_api_key
        )
        
        # 运行完整流程
        system.run_full_pipeline(
            pdf_paths=args.pdf_paths,
            queries=args.queries,
            output_dir=args.output_dir,
            force_rebuild=args.force_rebuild,
            evaluate=not args.no_evaluate,
            analyze_difficulty=not args.no_difficulty
        )
        
    except Exception as e:
        print(f"系统运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()