
# Survey Response Quality Analysis

This project provides a complete pipeline for analyzing the **quality of open-ended survey responses** using a combination of LLMs (Large Language Models) and classical ML techniques. It enables automatic flagging of low-quality, plagiarized, superficial, nonsensical, or out-of-context responses, making it a powerful tool for cleaning survey datasets and detecting response fraud (e.g., survey farms).

## 🧠 Motivation

Open-ended survey responses often suffer from issues like copy-pasting, gibberish text, or low-effort answers. Manually reviewing them is impractical at scale. This system automates the detection and scoring of such quality issues using LLMs and ML, offering numeric metrics and enabling downstream supervised model training for automated classification.

---

## 🔍 Project Features

* End-to-end preprocessing and cleaning of open-ended survey responses.
* Uses LLMs to compute:

  * **Plagiarism Score**
  * **Superficiality Score**
  * **QA Relevance Score**
  * **Out-of-Contextness**
  * **Non-Sensical Score**
* Computes:

  * **Response Time**
  * **Repetitiveness Score**
* Aggregates transformed metrics into a final feature dataset.
* Trains an ML model to predict response quality.
* Supports future extensions (e.g., intra-survey repetitiveness, translation support).

---

## 🗂️ Pipeline Overview

### 1. **Cleaning and Preprocessing**

![Cleaning Responses](./path/to/Cleaning%20Responses.png)

* **Input**: Survey responses + survey metadata (data dictionary).
* **Steps**:

  * Extract Open-Ended (OE) and Optional Questions.
  * Use LLM to assess question importance (weights).
  * Remove duplicate responses.
  * Handle missing fields with "Unanswered".
  * Translate responses if necessary.
  * Remove inter-survey repetitive responses.
  * Compute per-sample response duration.

---

### 2. **Model Training & Scoring**

![Model Training](./path/to/Model%20Training.png)

* **Input**: Cleaned response samples (`n x m` matrix).
* For each response:

  * If length > 5:

    * Run through LLM pipeline to compute:

      * Plagiarism Score
      * Superficiality Score
      * QA Relevance Score (via out-of-context and nonsensical check)
    * Weight these metrics appropriately.
  * Record response time and repetitiveness score.
* Aggregate per-sample metrics and build a numerical dataset.
* Train an ML model (e.g., Random Forest, XGBoost).
* Perform hyperparameter tuning and testing.

---

## 📊 Output

* `Transformed Sample Dataset`: Numerical scores per sample.
* `Trained ML Model`: Flags low-quality or suspicious responses.
* Optional: Visualizations and interpretability reports.

---

## 📁 Repository Structure

```
Survey-Analysis/
│
├── data/                   # Raw and preprocessed survey datasets
├── models/                 # Saved ML models
├── notebooks/              # Jupyter notebooks for each pipeline stage
├── utils/                  # Helper scripts (LLM scoring, cleaning)
├── configs/                # Configuration files for tuning models
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

---

## 🛠️ Dependencies

* Python ≥ 3.8
* OpenAI / HuggingFace Transformers
* scikit-learn
* pandas, numpy
* matplotlib / seaborn (for visualizations)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/Sanyam753/Survey-Analysis.git
   cd Survey-Analysis
   ```

2. Prepare your survey dataset:

   * Must include response matrix (`N x M`) and metadata.

3. Run preprocessing:

   ```bash
   python scripts/preprocess.py --input data/raw_survey.csv
   ```

4. Train the ML model:

   ```bash
   python scripts/train_model.py
   ```

5. Predict and evaluate:

   ```bash
   python scripts/evaluate.py --model models/final.pkl --data data/test.csv
   ```

---

## 📈 Future Scope

* Repetitive pattern detection within same survey
* Contextual similarity scoring between questions and answers
* Multilingual response handling and translation scoring
* Explainability features for model predictions

---

## 🙌 Acknowledgements

This project integrates concepts from survey quality analysis literature and the latest advancements in Large Language Models. Special thanks to contributors- Aniket Verma and Ishan Sharma and the open-source LLM community.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
