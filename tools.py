# PaperReader-"""
工具模块：为 Agent 提供论文分析能力
"""
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchInput(BaseModel):
    """搜索工具输入"""
    query: str = Field(description="搜索查询，关于论文内容的问题")
    top_k: int = Field(default=3, description="返回最相关的段落数")


class PaperSearchTool(BaseTool):
    """论文内容检索工具：在论文片段中搜索相关信息"""
    name: str = "PaperSearch"
    description: str = """在论文中搜索与查询最相关的段落。
    输入应为具体问题，如"论文使用了什么数据集"、"实验结果如何"。
    返回带页码和章节位置的原文片段。"""
    args_schema: Type[BaseModel] = SearchInput
    
    def __init__(self, chunks: List[dict]):
        super().__init__()
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self._build_index()
    
    def _build_index(self):
        """构建 TF-IDF 索引"""
        texts = [c['content'] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def _run(self, query: str, top_k: int = 3) -> str:
        """执行搜索"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                f"[第{chunk['page']}页 | {chunk['section']} | 相关度:{similarities[idx]:.3f}]\n"
                f"{chunk['content'][:500]}..."
            )
        
        return "\n\n---\n\n".join(results) if results else "未找到相关内容"


class SummarizeInput(BaseModel):
    """总结工具输入"""
    section: Optional[str] = Field(default=None, description="指定章节，如'abstract'、'method'")
    focus: str = Field(default="核心贡献", description="总结侧重点")


class PaperSummarizeTool(BaseTool):
    """论文总结工具：生成结构化摘要"""
    name: str = "PaperSummarize"
    description: str = """总结论文的指定部分或全文。
    可指定章节和关注重点，如'核心贡献'、'方法细节'、'实验结果'。
    输出严格基于原文，标注引用位置。"""
    args_schema: Type[BaseModel] = SummarizeInput
    
    def __init__(self, chunks: List[dict]):
        super().__init__()
        self.chunks = chunks
    
    def _run(self, section: Optional[str] = None, focus: str = "核心贡献") -> str:
        """执行总结"""
        if section:
            filtered = [c for c in self.chunks if c['chunk_type'] == section]
        else:
            filtered = self.chunks
        
        if not filtered:
            return f"未找到章节: {section}"
        
        # 拼接内容（简化版，实际应由 LLM 生成）
        content = "\n".join([c['content'][:1000] for c in filtered[:5]])
        return f"【{focus}总结】\n基于{len(filtered)}个段落：\n{content[:2000]}..."


class CompareInput(BaseModel):
    """对比工具输入"""
    paper2_path: str = Field(description="对比论文的PDF路径")
    aspect: str = Field(default="方法", description="对比维度：方法/实验/结论")


class PaperCompareTool(BaseTool):
    """论文对比工具：比较多篇论文"""
    name: str = "PaperCompare"
    description: str = """对比当前论文与另一篇论文在指定维度的异同。
    输入对比论文路径和对比维度（方法/实验/结论）。
    输出结构化对比表格。"""
    args_schema: Type[BaseModel] = CompareInput
    
    def _run(self, paper2_path: str, aspect: str = "方法") -> str:
        """执行对比（简化版）"""
        return f"对比功能：将当前论文与 {paper2_path} 在 '{aspect}' 维度对比。\n请先解析第二篇论文。"


class ExtractInput(BaseModel):
    """提取工具输入"""
    info_type: str = Field(description="提取类型：dataset/metric/model/author")


class PaperExtractTool(BaseTool):
    """信息提取工具：提取论文中的关键实体"""
    name: str = "PaperExtract"
    description: str = """从论文中提取结构化信息。
    支持提取：数据集(dataset)、评估指标(metric)、模型名称(model)、作者(author)。
    返回JSON格式结果，标注来源位置。"""
    args_schema: Type[BaseModel] = ExtractInput
    
    def __init__(self, chunks: List[dict]):
        super().__init__()
        self.chunks = chunks
    
    def _run(self, info_type: str) -> str:
        """执行提取"""
        # 简化实现：通过关键词匹配
        patterns = {
            'dataset': r'(?i)(dataset|数据集|benchmark|corpus)\s*[:\-]?\s*([A-Za-z0-9\-]+)',
            'metric': r'(?i)(accuracy|f1|bleu|rouge|precision|recall|em|metric|指标)',
            'model': r'(?i)(model|architecture|framework|proposed|ours)\s*[:\-]?\s*([A-Z][A-Za-z0-9\-]*)',
        }
        
        import re
        pattern = patterns.get(info_type, r'')
        results = []
        
        for chunk in self.chunks:
            matches = re.findall(pattern, chunk['content'])
            if matches:
                results.append({
                    'matches': matches[:3],
                    'source': f"第{chunk['page']}页-{chunk['section']}",
                    'context': chunk['content'][:200]
                })
        
        return f"提取结果 [{info_type}]:\n" + "\n".join([
            f"- {r['matches']} (来源: {r['source']})" for r in results[:10]
        ])
