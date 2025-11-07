"""
文档处理模块
负责PDF文档解析、向量化存储和检索
"""
import os
import pickle
from typing import List, Optional
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from llama_index.core import Settings
import chromadb
import nest_asyncio


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(
        self,
        llama_api_key: str = None,
        persist_dir: str = "./vector_store",
        cache_dir: str = "./document_cache"
    ):
        """
        初始化文档处理器
        
        Args:
            llama_api_key: LlamaParse API密钥
            persist_dir: 向量存储目录
            cache_dir: 文档缓存目录
        """
        # 应用异步支持
        nest_asyncio.apply()
        
        self.llama_api_key = llama_api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_api_key:
            raise ValueError("请设置LLAMA_CLOUD_API_KEY环境变量或传入llama_api_key参数")
        
        self.persist_dir = persist_dir
        self.cache_dir = cache_dir
        
        # 创建目录
        os.makedirs(persist_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化Chroma客户端
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.chroma_client.get_or_create_collection("communication")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 初始化分块器
        self.splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            include_metadata=True,
            include_prev_next_rel=True
        )
        
        self.index = None
        self.nodes = None
    
    def parse_documents(self, pdf_paths: List[str]) -> List:
        """
        解析PDF文档
        
        Args:
            pdf_paths: PDF文件路径列表
            
        Returns:
            解析后的文档列表
        """
        parser = LlamaParse(
            result_type="markdown",
            language="ch_sim",
            verbose=True,
            num_workers=4,
            include_page_number=True,
            parse_with_pymupdf=True,
            heading_detection=True,
            toc_detection=True,
            heading_detection_options={
                "min_text_size": 10,
                "max_link_density": 0.1,
                "require_numbered": False
            },
            metadata_inclusion=["page_label", "file_name", "page_number", "document_title", "section_title"],
            api_key=self.llama_api_key,
            extra_info={"source": "pdf_document"}
        )
        
        all_docs = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"警告: 文件不存在 {pdf_path}")
                continue
                
            documents = parser.load_data(pdf_path)
            print(f"解析完成: {pdf_path}，文档数量: {len(documents)}")
            all_docs.extend(documents)
        
        # 保存解析结果
        cache_file = os.path.join(self.cache_dir, "all_docs.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(all_docs, f)
        
        print(f"解析后的文档已保存到 {cache_file}")
        return all_docs
    
    def load_cached_documents(self) -> List:
        """
        加载缓存的文档
        
        Returns:
            缓存的文档列表
        """
        cache_file = os.path.join(self.cache_dir, "all_docs.pkl")
        if not os.path.exists(cache_file):
            raise FileNotFoundError("缓存文档不存在，请先解析文档")
        
        with open(cache_file, "rb") as f:
            all_docs = pickle.load(f)
        
        print("缓存文档加载成功")
        return all_docs
    
    def build_index(self, documents: List) -> None:
        """
        构建向量索引
        
        Args:
            documents: 文档列表
        """
        # 分块处理
        self.nodes = self.splitter.get_nodes_from_documents(documents)
        print(f"文档分块完成，节点数量: {len(self.nodes)}")
        
        # 构建索引
        self.index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model
        )
        
        # 持久化索引
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("向量索引构建并保存完成！")
    
    def load_index(self) -> None:
        """加载已存在的索引"""
        self.index = VectorStoreIndex.from_storage(self.storage_context)
        # 需要从索引中获取nodes信息
        if hasattr(self.index, 'docstore'):
            self.nodes = list(self.index.docstore.docs.values())
        print("向量索引加载完成！")
    
    def create_retrievers(self, vector_top_k: int = 10, bm25_top_k: int = 10):
        """
        创建检索器
        
        Args:
            vector_top_k: 向量检索返回数量
            bm25_top_k: BM25检索返回数量
            
        Returns:
            向量检索器和BM25检索器
        """
        if not self.index or not self.nodes:
            raise ValueError("请先构建或加载索引")
        
        # 向量检索器
        vector_retriever = VectorIndexRetriever(
            index=self.index, 
            similarity_top_k=vector_top_k
        )
        
        # BM25检索器
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes, 
            language="chinese", 
            similarity_top_k=bm25_top_k
        )
        
        return vector_retriever, bm25_retriever


class HybridRetriever:
    """混合检索器，结合向量检索和BM25检索"""
    
    def __init__(self, vector_retriever, bm25_retriever, vector_weight: float = 0.6):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            vector_weight: 向量检索权重
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
    
    def retrieve(self, query_bundle: QueryBundle) -> List:
        """
        执行混合检索
        
        Args:
            query_bundle: 查询包
            
        Returns:
            检索结果节点列表
        """
        # 向量检索
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # RRF（Reciprocal Rank Fusion）融合
        node_scores = {}
        
        # 向量检索得分
        for i, node in enumerate(vector_nodes):
            node_id = node.node.node_id
            score = self.vector_weight / (i + 1)
            node_scores[node_id] = node_scores.get(node_id, 0) + score
        
        # BM25检索得分
        for i, node in enumerate(bm25_nodes):
            node_id = node.node.node_id
            score = (1 - self.vector_weight) / (i + 1)
            node_scores[node_id] = node_scores.get(node_id, 0) + score
        
        # 按得分排序
        ranked_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回排序后的节点
        result_nodes = []
        node_dict = {node.node.node_id: node for node in vector_nodes + bm25_nodes}
        
        for node_id, _ in ranked_nodes:
            if node_id in node_dict:
                result_nodes.append(node_dict[node_id])
        
        return result_nodes