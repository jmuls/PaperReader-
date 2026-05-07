# PaperReader - 学术论文智能阅读助手

## 项目概述

基于 ReAct Agent 架构的论文阅读智能体，能够：
- 自动解析 PDF 论文结构（摘要、方法、实验、结论）
- 智能分块处理超长论文（50+页）
- 多步骤推理回答学术问题
- 严格引用约束，确保忠于原文

## 技术亮点

| 特性 | 实现 |
|------|------|
| ReAct 架构 | Thought → Action → Observation 循环推理 |
| 层次化记忆 | 章节级索引 + 段落级检索 |
| 引用溯源 | 每个事实标注页码和章节位置 |
| 超长论文支持 | 智能分块 + 重叠保持上下文连贯 |
| 工具调用 | 搜索/总结/提取/对比 四种工具 |

## 快速开始

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
python main.py
