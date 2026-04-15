#  Skew-Aware Distributed Vector Search (Harmony-Inspired)

This project implements a **Harmony-inspired distributed vector search system** for **network flow analytics under skewed workloads**.

It explores how **hybrid partitioning (vector + dimension)** improves:
- Throughput  
- Tail latency (P99)  
- Load balancing  

The system is evaluated on **CIC-IDS2017 network traffic data** using both **baseline and advanced pipelines**.

---

# Project Overview

Real-world network traffic is:
- Bursty  
- Highly skewed  
- Unevenly distributed  

Traditional distributed vector search systems suffer from:
- Load imbalance  
- High tail latency  
- Poor scalability under skew  

 This project evaluates whether **Harmony-style hybrid partitioning** can address these challenges.

---

# System Variants

We implement and compare **two main versions**:

## Version 1 (Primary — Recommended)
- Hybrid partitioning (vector + dimension)
- Early pruning
- Static configuration
- Produces best results

## Version 2 (Extended / Experimental)
- Adaptive repartitioning (cost-model based)
- Recall@K evaluation
- Load-aware tuning
- Demonstrates real-world challenges (overhead, tuning sensitivity)

---

# 📁 Project Structure

```
project-root/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── preprocess_cicids2017.py
│   ├── vector_baseline.py
│   ├── dimension_baseline.py
│   ├── harmony_hybrid.py        # Version 1
│   ├── harmony_pipeline.py      # Version 2
│   └── run_experiments.py
├── requirements.txt
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/Yateeka/ADCN-Project.git
cd ADCN-Project
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup (CIC-IDS2017)

## Option 1: Kaggle Download

```bash
kaggle datasets download -d chethuhn/network-intrusion-dataset -p data/raw
unzip data/raw/network-intrusion-dataset.zip -d data/raw/cicids2017
```

---

## Expected Structure

```
data/raw/cicids2017/
    *.csv
```

---

# 🔄 Step 1: Preprocess Dataset

```bash
python scripts/preprocess_cicids2017.py
```

## Output

```
data/processed/
├── base_vectors.npy
├── query_vectors.npy
├── queries_uniform.npy
├── queries_skewed.npy
├── base_labels.csv
├── query_labels.csv
```

---

# 📊 Step 2: Run Baselines

## Vector Partitioning

```bash
python scripts/vector_baseline.py --query-set uniform
python scripts/vector_baseline.py --query-set skewed
```

---

## Dimension Partitioning

```bash
python scripts/dimension_baseline.py --query-set uniform
python scripts/dimension_baseline.py --query-set skewed
```

---

# Step 3: Run Harmony Systems

## Version 1 (Recommended)

```bash
python scripts/harmony_hybrid.py --query-set skewed
```

Use this for:
- Final results  
- Report graphs  
- Presentation  

---

## Version 2 (Extended System)

### Full adaptive pipeline:

```bash
python scripts/harmony_pipeline.py --query-set skewed --compute-recall
```

### Disable adaptive (ablation):

```bash
python scripts/harmony_pipeline.py --query-set skewed --no-adaptive --compute-recall
```

---

# Step 4: Run Full Evaluation

```bash
python scripts/run_experiments.py
```

Outputs:
- Comparison table  
- Plots (throughput, latency, imbalance)  
- Evaluation graphs  

---

# Metrics

Each system reports:

- Throughput (QPS)  
- Mean latency  
- P95 / P99 latency  
- Load imbalance ratio  
- Pruning efficiency  
- Recall@K (Version 2)

---

# Key Insights

- Hybrid partitioning improves load balance and reduces latency  
- Load imbalance strongly correlates with tail latency  
- Adaptive systems require careful tuning  
- Simple baselines can outperform complex systems under certain workloads  

---

# Notes

- This is a **research prototype**
- Original Harmony system is implemented in C++ and not publicly available
- This implementation reproduces the **core algorithmic ideas in Python**
- Version 2 is experimental and may show degraded performance due to overhead

---

# Reproducibility

To reproduce main results:

```bash
python scripts/preprocess_cicids2017.py
python scripts/harmony_hybrid.py --query-set skewed
```

---

# Code Availability

The implementation of this project is available at:  
https://github.com/Yateeka/ADCN-Project

---

# Authors

- Yateeka Goyal  
- Apu Kumar Chakroborti  

---

# License

Academic use only

---

# Final Notes

- Use **Version 1 for main results**
- Version 2 demonstrates:
  - Adaptive system behavior  
  - Practical trade-offs  
