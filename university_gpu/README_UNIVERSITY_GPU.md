\# University GPU TTT-Discover Experiments



This folder contains my university GPU adaptation experiments for TTT-Discover.



\## Goal



The goal is to adapt TTT-Discover so that it can run on university GPU servers without relying on Tinker.



\## Planned stages



1\. Check Python, CUDA, and PyTorch environment on the university server.

2\. Run the official Section 4.4 single-cell denoising evaluator.

3\. Run a small local LoRA smoke test.

4\. Implement a tiny local TTT loop with a smaller LLM.

5\. Later adapt the framework to epidemic / dengue forecasting.



\## Notes



The original TTT-Discover training backend uses Tinker.



This university GPU version will try to replace the Tinker backend with:



\- Hugging Face Transformers

\- PEFT / LoRA

\- PyTorch optimizer

\- Local GPU generation and training

