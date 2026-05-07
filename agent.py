"""
ReAct Agent：论文阅读智能体的核心推理引擎
"""
import json
import re
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from .tools import PaperSearchTool, PaperSummarizeTool, PaperExtractTool
from .prompts import REACT_SYSTEM_PROMPT


class PaperReaderAgent:
    """论文阅读智能体"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,  # 低温度确保确定性
            max_tokens=2000
        )
        self.tools: List[BaseTool] = []
        self.chunks: List[Dict] = []
        self.agent_executor: Optional[AgentExecutor] = None
    
    def load_paper(self, chunks: List[Dict]):
        """加载论文分块，初始化工具"""
        self.chunks = chunks
        
        # 初始化工具
        self.tools = [
            PaperSearchTool(chunks),
            PaperSummarizeTool(chunks),
            PaperExtractTool(chunks),
        ]
        
        # 创建 ReAct Agent
        prompt = PromptTemplate.from_template(
            REACT_SYSTEM_PROMPT + """
            
工具描述：
{tools}

工具名称：{tool_names}

用户问题：{input}

{agent_scratchpad}
"""
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,  # 限制推理步数
            handle_parsing_errors=True
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """向 Agent 提问"""
        if not self.agent_executor:
            return {"error": "请先加载论文"}
        
        try:
            result = self.agent_executor.invoke({"input": question})
            return {
                "question": question,
                "answer": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "success": False
            }
    
    def summarize(self, focus: str = "核心贡献") -> str:
        """快速总结论文"""
        if not self.chunks:
            return "未加载论文"
        
        # 优先使用摘要和结论部分
        key_chunks = [
            c for c in self.chunks 
            if c.get('chunk_type') in ['abstract', 'conclusion', 'introduction']
        ]
        
        context = "\n\n".join([
            f"[第{c.get('page', '?')}页-{c.get('section', '?')}]\n{c['content'][:800]}"
            for c in key_chunks[:5]
        ])
        
        from .prompts import SUMMARIZE_PROMPT
        prompt = SUMMARIZE_PROMPT.format(chunks=context)
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def extract_key_info(self) -> Dict[str, List[str]]:
        """提取论文关键信息"""
        if not self.tools:
            return {}
        
        extract_tool = next(t for t in self.tools if t.name == "PaperExtract")
        
        info_types = ['dataset', 'metric', 'model']
        results = {}
        for info_type in info_types:
            try:
                result = extract_tool.run(info_type)
                results[info_type] = result
            except Exception as e:
                results[info_type] = f"提取失败: {e}"
        
        return results
