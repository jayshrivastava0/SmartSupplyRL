# SmartSupplyRL: Inventory Optimization with RL, GenAI & MCP

This project explores how reinforcement learning (RL) and generative AI (GenAI) can be combined to improve inventory planning in supply chain scenarios.

The goal is to train an RL agent that learns an optimal restocking policy from real retail demand data (using the M5 Forecasting dataset). The agent will aim to minimize costs while avoiding stockouts — effectively simulating the daily decision-making process of a supply planner.

To make the system more interactive and context-aware, I plan to integrate a lightweight open-source LLM (such as Meta’s LLaMA or Openweight) that can help simulate demand shifts, generate what-if scenarios, or explain policy behavior. These components will communicate using the **Model Context Protocol (MCP)**, which allows structured, secure connections between models and tools.


## Working Demo




https://github.com/user-attachments/assets/07e9ef6d-7306-4069-a934-e6b9a5af0f36




### Relevance to Blue Yonder

During my research, I found that Blue Yonder has been exploring the use of LLMs alongside supply chain decision engines. This project is an attempt to build a simplified version of that vision where a language model works alongside a reinforcement learning agent to support better planning through natural interaction and scenario understanding.

### Tools & Frameworks
- Python, PyTorch, Gymnasium
- PPO/DQN for RL
- M5 Forecasting dataset (Kaggle)
- Open-source LLM (LLaMA or something open weight)
- Model Context Protocol (MCP)

This is still a work in progress and will evolve as I add more components and connect the pieces.


