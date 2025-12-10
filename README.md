🌟 Demos Repository

A collection of hands-on, real-world AI and backend demo projects.
Each demo is fully contained in its own subfolder with documentation and runnable code.

<p align="center"> <img src="https://img.shields.io/badge/Powered_by-Python_3.11-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/FastAPI-Project-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/AI-Demos-orange?style=for-the-badge" /> </p>


📂 Project Index
1. 🤖 AI Chat Prefill Server

Folder: `ai-chat-prefill-server/`

A lightweight AI proxy server that provides:

    - OpenAI-style /v1/chat/completions endpoint
    - Free model fallback system (auto-detects which free model is available)
    - Payment metadata extractor via /v1/prefill
    - Writes structured output to data.csv
    - Clean FastAPI architecture + tests

➡️ Full details in:  
`ai-chat-prefill-server/README.md`


🧭 Repository Structure

```
demos/
│
├── ai-chat-prefill-server/
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   ├── tests/
│   │   └── test_chat_and_prefill_extended.py
│   └── data.csv
│
└── (More demo projects will be added…)
```



📄 License

Distributed under the **MIT License**.  
You are free to reuse, modify, and build upon these demos.
