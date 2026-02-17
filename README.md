---
title: SWE-Chatbot-Arena
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
hf_oauth: true
pinned: false
short_description: Chatbot arena for software engineering tasks
---

# SWE-Chatbot-Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering

Welcome to **SWE-Chatbot-Arena**, an open-source platform designed for evaluating software engineering-focused foundation models (FMs), particularly large language models (LLMs). SWE-Chatbot-Arena benchmarks models in iterative, context-rich workflows that are characteristic of software engineering (SE) tasks.

## Key Features

- **Multi-Round Conversational Workflows**: Evaluate models through extended, context-dependent interactions that mirror real-world SE processes.
- **RepoChat Integration**: Automatically inject repository context (issues, commits, PRs) into conversations for more realistic evaluations.
- **Advanced Evaluation Metrics**: Assess models using a comprehensive suite of metrics including:
  - **Traditional ranking metrics**: Elo ratings and win rates to measure overall model performance
  - **Network-based metrics**: Eigenvector centrality and PageRank to identify influential models in head-to-head comparisons
  - **Community detection metrics**: Newman modularity to reveal clusters of models with similar capabilities
  - **Consistency metrics**: Self-play match analysis to quantify model determinism and reliability
  - **Efficiency metrics**: Conversation efficiency index to measure response quality relative to length
- **Transparent, Open-Source Leaderboard**: View real-time model rankings across diverse SE workflows with full transparency.
- **Intelligent Request Filtering**: Employ `gpt-oss-safeguard-20b` as a guardrail to automatically filter out non-software-engineering-related requests, ensuring focused and relevant evaluations.

## Why SWE-Chatbot-Arena?

Existing evaluation frameworks (e.g. [LMArena](https://lmarena.ai)) often don't address the complex, iterative nature of SE tasks. SWE-Chatbot-Arena fills critical gaps by:

- Supporting context-rich, multi-turn evaluations to capture iterative workflows
- Integrating repository-level context through RepoChat to simulate real-world development scenarios
- Providing multidimensional metrics for nuanced model comparisons
- Focusing on the full breadth of SE tasks beyond just code generation

## How It Works

1. **Submit a Prompt**: Sign in and input your SE-related task (optional: include a repository URL for RepoChat context)
2. **Compare Responses**: Two anonymous models provide responses to your query
3. **Continue the Conversation**: Test contextual understanding over multiple rounds
4. **Vote**: Choose the better model at any point, with ability to re-assess after multiple turns

## Getting Started

### Prerequisites

- A [Hugging Face](https://huggingface.co) account

### Usage

1. Navigate to the [SWE-Chatbot-Arena platform](https://huggingface.co/spaces/SE-Arena/SWE-Chatbot-Arena)
2. Sign in with your Hugging Face account
3. Enter your SE task prompt (optionally include a repository URL for RepoChat)
4. Engage in multi-round interactions and vote on model performance

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Submit SE Tasks**: Share your real-world SE problems to enrich our evaluation dataset
2. **Report Issues**: Found a bug or have a feature request? Open an issue in this repository
3. **Enhance the Codebase**: Fork the repository, make your changes, and submit a pull request

## Privacy Policy

Your interactions are anonymized and used solely for improving SWE-Chatbot-Arena and FM benchmarking. By using SWE-Chatbot-Arena, you agree to our Terms of Service.

## Future Plans

- **Analysis of Real-World SE Workloads**: Identify common patterns and challenges in user-submitted tasks
- **Multi-Round Evaluation Metrics**: Develop specialized metrics for assessing model adaptation over successive turns
- **Expanded FM Coverage**: Include multimodal and domain-specific foundation models
- **Advanced Context Compression**: Integrate techniques like [LongRope](https://github.com/microsoft/LongRoPE) and [SelfExtend](https://github.com/datamllab/LongLM) to manage long-term memory in multi-round conversations

## Contact

For inquiries or feedback, please [open an issue](https://github.com/SE-Arena/SWE-Chatbot-Arena/issues/new) in this repository. We welcome your contributions and suggestions!

## Citation

Made with ❤️ for SWE-Chatbot-Arena. If this work is useful to you, please consider citing our vision paper:

```bibtex
@inproceedings{zhao2025se,
  title={SE Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering},
  author={Zhao, Zhimin},
  booktitle={2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge)},
  pages={78--81},
  year={2025},
  organization={IEEE}
}
```