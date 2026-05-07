"""
论文解析模块：提取结构、智能分块、保留上下文
"""
import re
import PyPDF2
from dataclasses import dataclass
from typing import List, Optional, Dict
import tiktoken


@dataclass
class PaperChunk:
    """论文分块，附带元数据"""
    content: str
    section: str          # 所属章节
    page: int             # 页码
    chunk_type: str       # abstract/introduction/method/experiment/conclusion/reference
    index: int            # 全局序号
    token_count: int      # token 数量


class PaperParser:
    """PDF 论文解析器"""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # 章节识别正则
        self.section_patterns = {
            'abstract': r'(?i)^\s*(abstract|摘要)',
            'introduction': r'(?i)^\s*\d*\s*(introduction|intro|引言)',
            'method': r'(?i)^\s*\d+\s*(method|methodology|approach|模型|方法)',
            'experiment': r'(?i)^\s*\d+\s*(experiment|evaluation|results|实验|评估)',
            'conclusion': r'(?i)^\s*\d+\s*(conclusion|discussion|结论|讨论)',
            'reference': r'(?i)^\s*(reference|bibliography|参考文献)',
        }
    
    def parse_pdf(self, pdf_path: str) -> List[PaperChunk]:
        """解析 PDF，返回带元数据的分块"""
        text = self._extract_text(pdf_path)
        sections = self._split_sections(text)
        chunks = []
        
        for section_name, section_text in sections:
            section_chunks = self._chunk_text(section_text, section_name)
            for i, chunk in enumerate(section_chunks):
                chunks.append(PaperChunk(
                    content=chunk,
                    section=section_name,
                    page=self._estimate_page(chunk),
                    chunk_type=self._classify_section(section_name),
                    index=len(chunks),
                    token_count=len(self.encoder.encode(chunk))
                ))
        
        return chunks
    
    def _extract_text(self, pdf_path: str) -> str:
        """提取 PDF 全文文本"""
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
        return text
    
    def _split_sections(self, text: str) -> List[tuple]:
        """按章节分割文本"""
        lines = text.split('\n')
        sections = []
        current_section = "preamble"
        current_text = []
        
        for line in lines:
            section_type = self._detect_section(line)
            if section_type and section_type != current_section:
                if current_text:
                    sections.append((current_section, '\n'.join(current_text)))
                current_section = section_type
                current_text = [line]
            else:
                current_text.append(line)
        
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))
        
        return sections
    
    def _detect_section(self, line: str) -> Optional[str]:
        """检测行是否为新章节标题"""
        for section_type, pattern in self.section_patterns.items():
            if re.match(pattern, line.strip()):
                return section_type
        return None
    
    def _chunk_text(self, text: str, section: str) -> List[str]:
        """按 token 数量分块，保持句子完整"""
        tokens = self.encoder.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            # 尝试在句子边界截断
            if end < len(tokens):
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoder.decode(chunk_tokens)
                # 找最后一个句号/换行
                last_break = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('。'),
                    chunk_text.rfind('\n')
                )
                if last_break > self.chunk_size * 0.5:
                    end = start + len(self.encoder.encode(chunk_text[:last_break+1]))
            
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - self.overlap  # 重叠保持连贯
        
        return chunks
    
    def _estimate_page(self, chunk: str) -> int:
        """估算页码（简化版）"""
        # 实际实现可通过 PDF 解析器获取精确页码
        return 0
    
    def _classify_section(self, section_name: str) -> str:
        """分类章节类型"""
        for type_name, pattern in self.section_patterns.items():
            if re.search(pattern, section_name, re.I):
                return type_name
        return "body"
