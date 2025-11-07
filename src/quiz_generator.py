"""
题目生成模块
负责基于检索结果生成选择题
"""
import os
from typing import List, Dict, Optional
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import QueryBundle
from llama_index.core import Settings


class QuizGenerator:
    """题目生成器"""
    
    def __init__(
        self,
        retriever,
        similarity_cutoff: float = 0.7,
        llm=None
    ):
        """
        初始化题目生成器
        
        Args:
            retriever: 检索器
            similarity_cutoff: 相似度阈值
            llm: 语言模型
        """
        self.retriever = retriever
        self.llm = llm or Settings.llm
        
        # 创建查询引擎
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self.llm,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )
        
        # 题目生成提示模板
        self.quiz_template = PromptTemplate(
            """根据以下文档片段，生成2道选择题，题目需基于片段内容，涵盖通信行业关键概念。每道题目需标注对应文档内容。输出格式如下：
    1. [选择题] 问题
       - A. 选项1
       - B. 选项2
       - C. 选项3
       - D. 选项4
       - 正确答案：X
       - 出处：{text}

    文档片段：
    {text}
    """
        )
        
        # 单题生成提示模板
        self.single_question_template = PromptTemplate(
            """根据以下文档片段，生成1道选择题，题目需基于片段内容，涵盖通信行业关键概念。输出格式如下：
    [选择题] 问题
    - A. 选项1
    - B. 选项2
    - C. 选项3
    - D. 选项4
    - 正确答案：X
    - 出处：{text}

    文档片段：
    {text}
    """
        )
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 10) -> List:
        """
        检索相关文档
        
        Args:
            query: 查询关键词
            top_k: 返回文档数量
            
        Returns:
            相关文档节点列表
        """
        query_bundle = QueryBundle(query_str=query)
        nodes = self.retriever.retrieve(query_bundle)
        return nodes[:top_k]
    
    def generate_question_from_node(self, node) -> str:
        """
        基于单个文档节点生成题目
        
        Args:
            node: 文档节点
            
        Returns:
            生成的题目文本
        """
        prompt = self.single_question_template.format(
            text=node.text
        )
        
        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            print(f"生成题目时出错: {e}")
            return None
    
    def generate_questions(
        self,
        query: str,
        num_questions: int = 6,
        start_index: int = 0,
        max_retries: int = 3
    ) -> List[str]:
        """
        生成指定数量的题目
        
        Args:
            query: 查询关键词
            num_questions: 生成题目数量
            start_index: 开始检索的索引
            max_retries: 最大重试次数
            
        Returns:
            生成的题目列表
        """
        # 检索相关文档
        nodes = self.retrieve_relevant_docs(query, top_k=num_questions + start_index + 5)
        
        if len(nodes) <= start_index:
            raise ValueError(f"检索到的文档数量不足，需要至少 {start_index + 1} 个文档")
        
        questions = []
        retry_count = 0
        
        # 从指定索引开始生成题目
        for i in range(start_index, min(len(nodes), start_index + num_questions + max_retries)):
            if len(questions) >= num_questions:
                break
                
            node = nodes[i]
            print(f"正在处理文档 {i+1}/{len(nodes)}...")
            
            question = self.generate_question_from_node(node)
            
            if question and self._validate_question_format(question):
                questions.append(question)
                print(f"成功生成题目 {len(questions)}/{num_questions}")
            else:
                print(f"文档 {i+1} 生成失败，跳过")
                retry_count += 1
                
                if retry_count >= max_retries:
                    print("达到最大重试次数，停止生成")
                    break
        
        if len(questions) < num_questions:
            print(f"警告: 只生成了 {len(questions)} 道题目，少于要求的 {num_questions} 道")
        
        return questions
    
    def _validate_question_format(self, question_text: str) -> bool:
        """
        验证题目格式是否正确
        
        Args:
            question_text: 题目文本
            
        Returns:
            格式是否正确
        """
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
            if element not in question_text:
                return False
        
        return True
    
    def save_quiz_to_file(
        self,
        questions: List[str],
        output_file: str,
        title: str = "通信文档问卷"
    ) -> None:
        """
        保存题目到文件
        
        Args:
            questions: 题目列表
            output_file: 输出文件路径
            title: 问卷标题
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            
            for i, question in enumerate(questions, 1):
                f.write(f"## 题目 {i}\n{question}\n\n")
        
        print(f"问卷生成完成，保存至 {output_file}")
    
    def generate_batch_quizzes(
        self,
        queries: List[str],
        questions_per_query: int = 3,
        output_prefix: str = "quiz"
    ) -> Dict[str, List[str]]:
        """
        批量生成题目
        
        Args:
            queries: 查询关键词列表
            questions_per_query: 每个查询生成的题目数量
            output_prefix: 输出文件前缀
            
        Returns:
            查询到题目的映射字典
        """
        all_questions = {}
        
        for query in queries:
            print(f"\n正在为查询 '{query}' 生成题目...")
            
            try:
                questions = self.generate_questions(
                    query=query,
                    num_questions=questions_per_query
                )
                all_questions[query] = questions
                
                # 保存单个查询的题目
                filename = f"{output_prefix}_{query.replace(' ', '_')}.md"
                self.save_quiz_to_file(
                    questions=questions,
                    output_file=filename,
                    title=f"通信文档问卷 - {query}"
                )
                
            except Exception as e:
                print(f"为查询 '{query}' 生成题目时出错: {e}")
                all_questions[query] = []
        
        # 保存合并的题目
        if all_questions:
            all_questions_list = []
            for query, questions in all_questions.items():
                all_questions_list.extend(questions)
            
            self.save_quiz_to_file(
                questions=all_questions_list,
                output_file=f"{output_prefix}_all.md",
                title="通信文档问卷 - 全部题目"
            )
        
        return all_questions


def get_quiz_config() -> dict:
    """获取题目生成配置"""
    return {
        "similarity_cutoff": 0.7,
        "default_num_questions": 6,
        "default_start_index": 0,
        "max_retries": 3
    }