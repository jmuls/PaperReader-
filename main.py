#!/usr/bin/env python3
"""
PaperReader - 学术论文智能阅读助手
使用 ReAct Agent 架构，支持超长论文处理
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from paperreader.parser import PaperParser
from paperreader.agent import PaperReaderAgent


def main():
    """主程序"""
    print("=" * 60)
    print("📄 PaperReader - 学术论文智能阅读助手")
    print("=" * 60)
    
    # 配置 API（请替换为你的实际配置）
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL = os.getenv("MODEL_NAME", "gpt-4o")
    
    # 1. 解析论文
    pdf_path = input("请输入论文PDF路径（或按Enter使用示例）: ").strip()
    if not pdf_path:
        # 创建示例论文文本（演示用）
        print("使用示例论文...")
        sample_text = create_sample_paper()
        parser = PaperParser(chunk_size=1500, overlap=150)
        # 模拟解析结果
        chunks = simulate_parse(sample_text, parser)
    else:
        if not os.path.exists(pdf_path):
            print(f"错误：文件不存在 {pdf_path}")
            return
        
        print("正在解析论文...")
        parser = PaperParser(chunk_size=2000, overlap=200)
        chunks = parser.parse_pdf(pdf_path)
        chunks = [{"content": c.content, "section": c.section, 
                   "page": c.page, "chunk_type": c.chunk_type} 
                  for c in chunks]
    
    print(f"✅ 论文解析完成，共 {len(chunks)} 个片段")
    
    # 2. 初始化 Agent
    print("正在初始化智能体...")
    agent = PaperReaderAgent(API_KEY, BASE_URL, MODEL)
    agent.load_paper(chunks)
    print("✅ 智能体就绪")
    
    # 3. 交互式问答
    print("\n" + "=" * 60)
    print("交互模式：输入问题，或输入 'summary' 总结，'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n🤔 你的问题: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("再见！")
                break
            
            if question.lower() == 'summary':
                print("\n📋 生成论文总结...")
                summary = agent.summarize()
                print(summary)
                continue
            
            if not question:
                continue
            
            print("\n🧠 正在思考...")
            result = agent.ask(question)
            
            if result.get("success"):
                print(f"\n✅ 回答:")
                print(result["answer"])
                
                # 显示推理过程
                steps = result.get("intermediate_steps", [])
                if steps:
                    print(f"\n🔍 推理过程 ({len(steps)} 步):")
                    for i, step in enumerate(steps, 1):
                        print(f"  步骤{i}: {str(step)[:100]}...")
            else:
                print(f"\n❌ 错误: {result.get('error', '未知错误')}")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def create_sample_paper() -> str:
    """创建示例论文文本（演示用）"""
    return """
Attention Is All You Need

Abstract
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train.

1 Introduction
Recurrent neural networks, long short-term memory and gated recurrent neural
networks have been firmly established as state of the art approaches in sequence
modeling and transduction problems such as language modeling and machine
translation. Numerous efforts have since continued to push the boundaries of
recurrent language models and encoder-decoder architectures.

2 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder
architecture. Here, the encoder maps an input sequence of symbol representations
(x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn).
Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols
one element at a time.

2.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each
layer has two sub-layers. The first is a multi-head self-attention mechanism,
and the second is a simple, position-wise fully connected feed-forward network.

3 Experiments
We trained our models on the WMT 2014 English-German dataset consisting of
about 4.5 million sentence pairs. On the WMT 2014 English-to-German translation
task, our big transformer model achieved a BLEU score of 28.4, improving
over the existing best results, including ensembles, by over 2.0 BLEU.

4 Conclusion
In this work, we presented the Transformer, the first sequence transduction model
based entirely on attention, replacing the recurrent layers most commonly used
in encoder-decoder architectures with multi-headed self-attention.
"""


def simulate_parse(text: str, parser: PaperParser) -> List[dict]:
    """模拟解析结果"""
    sections = parser._split_sections(text)
    chunks = []
    
    for section_name, section_text in sections:
        section_chunks = parser._chunk_text(section_text, section_name)
        for i, chunk_text in enumerate(section_chunks):
            chunks.append({
                "content": chunk_text,
                "section": section_name,
                "page": i + 1,
                "chunk_type": parser._classify_section(section_name),
            })
    
    return chunks


if __name__ == "__main__":
    main()
