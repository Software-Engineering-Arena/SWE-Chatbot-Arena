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

# SWE-Chatbot-Arena

An open-source arena for evaluating LLMs on **real software engineering tasks** — multi-turn, context-rich, and repo-aware.

Unlike general-purpose arenas (e.g., LMArena), this platform focuses on iterative SE workflows: debugging, code review, refactoring, and more.

## How It Works

1. Sign in with your [Hugging Face](https://huggingface.co) account
2. Enter an SE task (optionally include a repo URL for automatic context injection via **RepoChat**)
3. Two anonymous models respond — compare them over multiple rounds
4. Vote for the better model

Try it: [SWE-Chatbot-Arena on HF Spaces](https://huggingface.co/spaces/SE-Arena/SWE-Chatbot-Arena)

## Key Capabilities

- **RepoChat** — injects repo context (issues, commits, PRs) into conversations
- **Multi-round evaluation** — tests contextual understanding across turns
- **Rich metrics** — Elo, PageRank, eigenvector centrality, modularity clustering, self-play consistency, efficiency index
- **Guardrails** — `gpt-oss-safeguard-20b` filters non-SE requests

## Contributing

Submit SE tasks, report bugs, or send PRs. [Open an issue](https://github.com/SE-Arena/SWE-Chatbot-Arena/issues/new) to get started.

## Citation

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
