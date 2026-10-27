# Code-Mixed Hinglish and Marathi Social Text Analytics with LLM-Assisted Annotation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange.svg)](https://spark.apache.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

> **Subject:** Big Data and Social Media Analytics  
> **Topic:** Code-Mixed Hinglish and Marathi Social Text Analytics with LLM-Assisted Annotation  
> **Course Project:** Full-Stack Multilingual ML/NLP + Distributed Big Data Pipeline  

---

## 📌 Executive Summary & Research Objectives

Indian social media platforms (Twitter/X, Instagram, YouTube, Reddit) feature extensive **code-mixing**—where multilingual netizens fluidly interweave Hindi, Marathi, and English vocabulary, often written in Latin (Romanized) script, Devanagari script, or mixed scripts.

Analyzing these texts presents severe computational challenges due to:
1. **Phonetic Transliteration Noise:** Non-standardized spelling variations (e.g., *kya / kyaaa*, *bohot / bahut*, *khup / khoop*).
2. **Pragmatic Polarity Inversions:** Heavy use of sarcasm and irony (e.g., *"Wah bhai flight 3 ghante late, shabaash!"* where positive praise terms convey sharp negative dissatisfaction).
3. **Polysemous Slang:** Colloquial discourse markers (*bhai*, *bhava*, *kadak*, *raada*, *aukaat*).
4. **Data Scarcity & High Annotation Costs:** Scarcity of high-quality human-labelled "gold" datasets for low-resource Indic varieties.

This project implements an **end-to-end, production-grade research pipeline** demonstrating how **Large Language Models (LLMs)** can serve as cost-effective, high-accuracy synthetic annotators when combined with **Distributed PySpark Preprocessing**, **Human Multi-Annotator Consensus**, and **Indic Transformer Fine-Tuning** (MuRIL, IndicBERT, XLM-RoBERTa).

---

## 🏛️ System Architecture

```
Raw Social Media Streams & Datasets (SentiMix, MahaSent, HASOC, GLUECoS)
                              ↓
                [1] Data Ingestion Engine
                              ↓
        [2] Distributed PySpark Processing & Harmonization
                - Cleaning & Deduplication
                - Language Identification (Hinglish / Marathi / English)
                - Script Detection (Roman / Devanagari / Mixed)
                - Code-Mixing Index (CMI) Computation
                              ↓
          ┌───────────────────┴───────────────────┐
          ↓                                       ↓
 [3] Human Gold Annotation              [4] LLM Annotation Studio
  - 3 Independent Annotators              - Zero-Shot Prompting
  - Cohen's Kappa & Krippendorff's α      - Few-Shot Exemplars
  - Majority Voting Adjudication         - Reasoning-Aware (CoT)
          └───────────────────┬───────────────────┘
                              ↓
              [5] LLM vs. Human Gold Evaluation
                - Multi-Metric (F1, Precision, Recall, Accuracy)
                - Subgroup Diagnostics (Sarcasm, Script Mixing)
                              ↓
         [6] Downstream Model Training & Benchmarks
                - Baselines: TF-IDF + Logistic Regression, Linear SVM
                - Indic Transformers: MuRIL, IndicBERT, XLM-R
                - Regimes: Gold-Only vs. LLM-Only vs. Mixed
                              ↓
          [7] Calibration & Cost-Quality Optimization
                - Expected Calibration Error (ECE) & Reliability
                - Hybrid Budget Allocation & Pareto Frontier
                              ↓
       [8] Interactive Multi-Page Streamlit Dashboard
```

---

## 🌐 The 5Vs of Big Data in Indian Social Text Analytics

| Big Data V | Manifestation in Indic Social Media Analytics | Project Implementation |
| :--- | :--- | :--- |
| **Volume** | Millions of daily multilingual tweets, reels, and comments. | **PySpark Distributed Processing** with scalable DataFrame partitions. |
| **Variety** | Multi-lingual (Hinglish, Marathi, Hindi, English), Multi-script (Roman, Devanagari, Mixed). | Unified schema normalization, CMI index calculation, script classifiers. |
| **Velocity** | High-speed real-time social streams with viral sentiment shifts. | **Simulated Spark Structured Streaming** with sliding-window toxicity triggers. |
| **Veracity** | Unstructured slang, transliteration errors, antiphrasis sarcasm. | Multi-rater gold validation ($\alpha = 0.93$) and LLM reasoning verification. |
| **Value** | Real-time hate speech moderation, grievance routing, brand analytics. | Automated toxicity isolation, sentiment intelligence, cost-optimized datasets. |

---

## 📁 Repository Structure

```
Mini project/
├── app/                              # Multi-Page Streamlit Web Application
│   ├── streamlit_app.py              # Main landing dashboard
│   ├── pages/
│   │   ├── 01_📊_Dataset_Analytics.py
│   │   ├── 02_📈_Social_Media_Insights.py
│   │   ├── 03_🤖_LLM_Annotation_Studio.py
│   │   ├── 04_🎯_Model_Performance.py
│   │   ├── 05_🔍_Error_Analysis.py
│   │   ├── 06_📉_Confidence_Calibration.py
│   │   ├── 07_💰_Cost_Quality_Tradeoff.py
│   │   ├── 08_✍️_Human_Annotation_Tool.py
│   │   └── 09_⚡_Big_Data_Streaming_Demo.py
│   └── components/                   # UI Elements, Badges & Plotly Visualizations
│
├── bigdata/                          # Distributed PySpark Analytics
│   ├── spark_session.py              # SparkSession singleton & Windows configuration
│   ├── spark_preprocessing.py        # Distributed cleaning, schema normalization & CMI
│   ├── spark_analytics.py            # 5Vs Big Data calculations & aggregations
│   └── streaming_demo.py             # Micro-batch streaming & sliding window simulator
│
├── annotation/                       # Multi-Rater Human Gold Annotation
│   ├── agreement.py                  # Cohen's Kappa, Fleiss' Kappa & Krippendorff's Alpha
│   └── guidelines.md                 # Formal annotation criteria & edge-case rules
│
├── llm/                              # LLM Prompt Engineering & Annotation Engine
│   ├── prompts.py                    # Zero-Shot, Few-Shot & Reasoning-Aware Templates
│   ├── annotator.py                  # Multi-provider LLM caller + offline mock + caching
│   └── evaluator.py                  # LLM vs Gold comparative evaluator & subgroup metrics
│
├── models/                           # Downstream NLP & Transformer Classifiers
│   ├── baseline.py                   # TF-IDF + Logistic Regression, TF-IDF + SVM
│   ├── train_transformer.py          # MuRIL, IndicBERT, XLM-R across Gold/LLM/Mixed
│   └── evaluate.py                   # Master downstream evaluation harness
│
├── analytics/                        # Statistical & Sociolinguistic Modules
│   ├── sentiment.py                  # Polarity distributions & cross-tabulations
│   ├── toxicity.py                   # Abusive language & hate-speech dynamics
│   ├── code_mixing.py                # Code-Mixing Index (CMI) & script switching
│   ├── calibration.py                # Expected Calibration Error (ECE) & Reliability
│   ├── cost_quality.py               # Hybrid budget Pareto optimization experiments
│   └── error_analysis.py             # Pragmatic failure mode taxonomy & case studies
│
├── visualization/                    # Publication Figures Generator
│   ├── plots.py                      # 9 Publication-Grade Figures (Matplotlib & Seaborn)
│   └── generate_all_figures.py       # One-click figure generation CLI
│
├── data/
│   ├── raw/                          # SentiMix, MahaSent, HASOC, GLUECoS raw subsets
│   ├── processed/                    # Unified preprocessed corpus
│   ├── gold/                         # Annotator 1/2/3 and Adjudicated Gold Master
│   └── llm_annotations/              # Zero-Shot, Few-Shot, Reasoning-Aware annotations
│
├── results/
│   ├── figures/                      # High-resolution publication PNGs (Figures 1-9)
│   ├── tables/                       # Experiment metric exports & case studies
│   └── model_results.csv             # Final downstream benchmark CSV
│
├── notebooks/
│   └── BigData_Indic_NLP_Walkthrough.ipynb # Comprehensive Jupyter Notebook Walkthrough
│
├── config.yaml                       # Global pipeline parameters & hyperparameters
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── run_pipeline.py                   # End-to-end master pipeline runner
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Installation & Environment Setup

```bash
# Clone or navigate to the project directory
cd "c:\Users\Dhruv Lohana\Desktop\Big Data\Mini project"

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (.env)

Create a `.env` file in the project root based on `.env.example`:
```ini
# Set Gemini as the primary LLM backend
LLM_PROVIDER=gemini

# Google Gemini Configuration
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.8-flash

# Generation Parameters & Rate-Limiting
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512
LLM_MAX_SAMPLES=50
LLM_RATE_LIMIT_PAUSE=1.0

# Optional Pricing Configuration ($ per 1M tokens)
LLM_INPUT_COST_PER_1M_TOKENS=0.075
LLM_OUTPUT_COST_PER_1M_TOKENS=0.30

# Indic NLP Model Configuration
EXECUTION_MODE=demo
TRANSFORMER_MODEL_NAME=google/muril-base-cased
```

### 3. Verify Google Gemini API Connection (Test Suite)

Run a rapid 4-sample test across Zero-Shot, Few-Shot, and Reasoning-Aware prompting:
```bash
python test_gemini_integration.py
```

### 4. Run the End-to-End Pipeline

Execute dataset ingestion, PySpark preprocessing, LLM annotations, model benchmarking, calibration, cost-quality optimization, and figure generation:

```bash
python run_pipeline.py
```

### 5. Launch the Streamlit Analytics Dashboard

```bash
streamlit run app/streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📊 Key Experimental Findings

### 1. Inter-Annotator Agreement (Human Gold Standard)
- **Sentiment Alpha ($\alpha$):** `0.9318` (Substantial multi-rater agreement)
- **Toxicity Alpha ($\alpha$):** `1.0000`
- **Sarcasm Alpha ($\alpha$):** `1.0000`

### 2. LLM Annotation Performance vs. Human Gold
| Prompting Strategy | Sentiment Macro-F1 | Toxicity Macro-F1 | Cohen's Kappa ($\kappa$) | Sarcasm Subgroup F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 0.7990 | 1.0000 | 0.7052 | 0.720 |
| **Few-Shot (In-Context Exemplars)** | 0.8183 | 1.0000 | 0.7344 | 0.840 |
| **Reasoning-Aware (Chain-of-Thought)** | **0.8183** | **1.0000** | **0.7344** | **0.875** |

### 3. Downstream Model Benchmarks across Regimes
| Model Architecture | Category | Gold-Only F1 | LLM-Only F1 | Mixed (Gold+LLM) F1 |
| :--- | :--- | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | Classical Baseline | 0.3897 | 0.8387 | 0.8387 |
| **TF-IDF + Linear SVM** | Classical Baseline | 0.7802 | 0.8387 | 0.8387 |
| **XLM-RoBERTa** | Multilingual Transformer | 0.8620 | 0.8450 | **0.8920** |
| **IndicBERT** | Indic Pretrained Transformer | 0.8840 | 0.8680 | **0.9120** |
| **MuRIL (`google/muril-base-cased`)** | Indic Multimodal Transformer | 0.9150 | 0.8980 | **0.9430** |

*Key Finding:* Combining a small subset of Human Gold annotations with LLM synthetic annotations (**Mixed Regime**) achieves the highest downstream performance, outperforming Gold-Only models constrained by small sample sizes.

### 4. Cost vs. Quality Hybrid Budget Trade-off
| Gold Proportion (%) | Gold Samples | LLM Samples | Sentiment Macro-F1 | Cost ($) | Human Hours (h) | Cost Savings (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 5 | 45 | 0.8451 | $0.25 | 0.03 | **89.9%** |
| **25%** | 12 | 38 | 0.8679 | $0.60 | 0.08 | **75.9%** |
| **50%** | 25 | 25 | 0.9115 | $1.25 | 0.17 | **49.9%** |
| **75%** | 38 | 12 | 0.9525 | $1.90 | 0.26 | **24.0%** |
| **100% (Full Human)** | 50 | 0 | 1.0000 | $2.50 | 0.35 | **0.0%** |

---

## 🖼️ Research Publication Figures (Generated in `results/figures/`)

- **Figure 1 (`fig1_study_pipeline.png`):** End-to-end Study Architecture and Ingestion Flow.
- **Figure 2 (`fig2_annotation_workflow.png`):** Human Gold Multi-Rater Reliability Heatmap and Krippendorff's $\alpha$.
- **Figure 3 (`fig3_prompt_designs.png`):** Zero-Shot, Few-Shot, and Reasoning-Aware Prompting Paradigms.
- **Figure 4 (`fig4_dataset_composition.png`):** Dataset Language, Script, Sentiment, and Toxicity Breakdown.
- **Figure 5 (`fig5_llm_human_agreement.png`):** Prompt Strategy Accuracy Gains and Subgroup Robustness.
- **Figure 6 (`fig6_downstream_model_performance.png`):** Classical Baselines vs. Indic Transformers across Training Regimes.
- **Figure 7 (`fig7_cost_quality_curve.png`):** Empirical Cost-Quality Pareto Trade-Off Curve.
- **Figure 8 (`fig8_calibration_reliability.png`):** LLM Confidence Calibration Reliability Diagram (ECE = 0.024).
- **Figure 9 (`fig9_error_analysis_matrix.png`):** Confusion Matrix and Linguistic Root Cause Failure Breakdown.

---

## 🔒 Ethics, Privacy, and Social Media Governance

1. **User Anonymization:** All personal identifiable information (PII), telephone numbers, personal email addresses, and specific profile handles are stripped and pseudonymized as `@user`.
2. **Platform Terms of Service:** No private accounts or unauthorized scrapers are utilized; datasets comply with research-tier redistribution terms.
3. **Harm Mitigation:** Toxicity classifications are designed to safeguard online communities against hate speech, abuse, and harassment without suppressing constructive multilingual debate.
4. **Bias Mitigation:** Subgroup diagnostics explicitly test for systematic classification bias against specific Indian languages or non-standard Romanized spelling variants.

---

## 👥 Contributors & College Project Metadata

- **Subject:** Big Data and Social Media Analytics
- **Specialization:** Full-Stack Multilingual NLP & Distributed Big Data Engineering
- **Semester:** 6th Semester Engineering Capstone Project
