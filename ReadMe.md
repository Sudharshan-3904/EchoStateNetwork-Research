# Echo State Network Research Project

This project is a research and prototyping environment for experimenting with Echo State Networks (ESNs) for natural language processing, specifically text generation, token prediction, and chatbot functionalities. It includes training scripts, inference modules, scraping utilities, and legacy experimental code.

---

## Table of Contents

- Objectives [ #Objectives ] Hi
- Project Structure
- Setup Instructions
- Usage Guide
- Dataset Information
- Legacy Code
- Research Findings
- Future Works
- Acknowledgments

---

## Objectives

- To build different variant of ESN
- To apply ESN to varous Hardware ( CPU, GPU, etc. )
- To validate the use of ECN for a chatbot
- To explore other applications for ESN's

---

## Project Structure

The files have been organized into the following structure for clarity and maintainability:

```bash
.
├──src/
│   ├── inference/
│   │   ├── chat_with_combined_model.py
│   │   ├── chat_with_combined_model_gpu.py
│   │   └── interactive_chatbot.py
│   ├── models/
│   │   └── esn_model.py
│   ├── training/
│   │   ├── train_next_token_gpu.py
│   │   └── train_with_webscraping_gpu.py
│   ├── utils/
│   │   └── combine_multiple_esn.py
│   ├── webscraping/
│   │   ├── web_scrape.py
│   │   └── web_token_scraping.py
│
├── legacy/
│   ├── esn_chat.py
│   ├── scrape_train.py
│   ├── combining_pkls/
│   │   ├── comb_esn_pkl.py
│   │   ├── comb_vect_pkl.py
│   │   ├── combine_pkl.py
│   │   └── vect_combine_pkl.py
│   └── wikitrain/
│       ├── esn_gpu_full.py
│       ├── esn_train.py
│       ├── esn_train_gpu.py
│       ├── esn_train_npu.py
│       └── esn_train_raw.txt
│
├── data/
│   ├── scraped_websites.txt
│   └── urls.csv
│
├── docs/
│   └── architecture.md
│
├── logs/
│   ├── batch_progress.txt
│   ├── batch_times.csv
│   └── error_log.txt
│
├── tests/
│   └── test_main.py
│
└── requirements.txt
```

---

## Setup Instructions

- Requirements:

  - Python 3.8 or above
  - Compatible with Raspberry Pi 3 (limited model sizes and performance constraints)

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Optional for GPU training (not available on Raspberry Pi):

  - torch with CUDA
  - sentence-transformers

---

## Usage Guide

**Train ESN with Web Scraping:**

```bash
python src/training/train_with_webscraping_gpu.py
```

**Train ESN on Token Prediction:**

```bash
python src/training/train_next_token_gpu.py
```

**Run Chatbot with Combined Model (GPU):**

```bash
python src/inference/chat_with_combined_model_gpu.py
```

**Interactive Terminal Chatbot:**

```bash
python src/inference/interactive_chatbot.py
```

---

## Dataset Information

- `data/urls.csv`: List of URLs used for scraping.
- `data/scraped_websites.txt`: Raw text collected from web scraping sessions.

Web scraping scripts are located in `src/webscraping/`.

---

## Legacy Code

The `legacy/` directory contains older versions and experimental scripts that are not actively maintained but may be helpful for reference or revival.

- `esn_chat.py` — basic chatbot prototype.
- `wikitrain/` — initial ESN training tests on Wikipedia datasets.
- `combining_pkls/` — experiments for merging model weights and vector pickles.

---

## Research Findings

- An ESN can handle small to mid term dependencies
- An ESN cannot hanlde long term dependecies required for Chatot like applications
- It can be used to assist a chatbot for simple tasks based on small substring
- A model with 1000 nodes in reservoir is suffcient for modeling short term dependencies
- ESN's cannot handle large vocabularies
- ESN properties have the following implications:

  | Property        | Purpose             | Description                                | Optimal value ( Experimental )    |
  | --------------- | ------------------- | ------------------------------------------ | --------------------------------- |
  | input_size      | Interface Parameter | To receive and feed to reservoir           | `Based on input`                  |
  | reservoir_size  | Model Definition    | Larger increases dependency lifetime       | 1000 ( mid term dependencies )    |
  | output_size     | Interface Parameter | To map the reservoir to outptu             | `Based on application`            |
  | spectral_radius | Fine Tuning         | How far spread out the model is            | 0.95                              |
  | sparsity        | Fine Tuning         | How common the connection b/w two nodes is | 0.1 ( 10% chance for connection ) |
  | input_scaling   | Fine Tuning         | Adjusts inputs to pass to reservoir        | 1.0                               |
  | leaking_rate    | Fine Tuning         | Impact network stability                   | 1.0                               |

---

## Future Works

Echo State Networks are best suited for other input types and thus have applicaitons in other fields. To further continue further, ESN can be applied in applicaitons and the fields mentioned below:

- Signal processign and signal based controlling
- Time series prediciton (Stock Price Prediciton, etc. )
- Potentially Speeck Recognition
- Financial data modelling

---

## Acknowledgments

This Research was conducted as an passion based exploration into the workings and applications of ESN primarily in Chatbot like applications. The general idea for the model architecture was designed based on papers published on the topic. A few key papers are provided under `docs`.

---
