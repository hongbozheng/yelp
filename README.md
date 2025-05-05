# 📊 CS412 Final Project – Mining User Behavior & Review Helpfulness on Yelp

## 🧠 Overview

This project explores real-world data mining techniques on the **Yelp Academic Dataset**, applying concepts learned in **CS412: Introduction to Data Mining**. The pipeline performs:

* 🔄 Data Preprocessing & Feature Engineering
* 📊 Frequent Pattern Mining
* 📈 User Clustering
* 🎯 Review Helpfulness Classification

The analysis targets discovering **reviewer behavior patterns**, segmenting users by activity, and **predicting helpful reviews** — all grounded in scalable data mining practices.

---

## 📦 Dataset

We use a filtered subset of the [Yelp Open Dataset](https://www.yelp.com/dataset) with `preprocess.py`:

* `review.json`: Review content, star ratings, feedback
* `business.json`: Business metadata including categories
* `user.json`: User-level metadata (review count, avg stars, etc.)
* `checkin.json`, `tip.json`: Optional behavioral metadata

We filter to:

* ✅ Top 10 cities by total review count
* ✅ Top categories (e.g., Restaurants, Bars, Cafes)
* ✅ Active users (≥ 20 reviews)
* ✅ Reviews after 2019-01-01

---

## 🔄 Phase 1: Data Preprocessing & Feature Engineering

### ✅ Key Steps

* Filter and join `review`, `business`, and `user` datasets
* Extract review metadata: length, usefulness, date
* Create binary category flags (e.g., `%Bars`, `%Sushi Bars`)
* Add user statistics: `avg_stars`, `fans`, `review_count`
* Add behavioral metadata: `tip_count`, `checkins`, `review_variance`

### 💾 Output

* A `df` containing structured features per review merged and averaged with user features

---

## 📊 Phase 2: Frequent Pattern Mining

We mine **frequent co-occurring business categories** and interpret user tastes.

### ✅ Experiments

1. **Compare helpful vs. non-helpful reviews**
   → What category patterns lead to helpful feedback?

2. **City-specific pattern discovery**
   → What makes Las Vegas reviews different from Tampa?

3. **Constraint-based mining**
   → Only mine patterns containing keywords like `'Bars'`, `'Sushi Bars'`

4. **Top-k diverse itemsets**
   → Reduce redundancy using Jaccard distance + clustering

5. **Visual Summaries**
   → Sankey plot, Co-occurrence graph, Lift heatmap

---

## 📈 Phase 3: User Clustering

We cluster users by their **review behavior**.

### ✅ Features Engineered

* `avg_stars`, `review_count`, `star_variance`
* `% category engagement` (`%Bars`, `%Cafes`, etc.)
* `review_consistency`, `tip_count`, `checkin_count`
* `unique_businesses`, `active_years`

### ✅ Methods

1. **K-Means**
2. **Hierarchical Clustering (Agglomerative + Dendrogram)**
3. **DBSCAN (density-based)**

### 📊 Evaluation

* Silhouette scores
* Cluster centroids
* PCA/t-SNE visualization
* Outlier detection (DBSCAN `-1` labels)

---

## 🎯 Phase 4: Review Helpfulness Classification

We predict whether a review is **helpful** (i.e., `useful ≥ 2`, `avg_useful ≥ 1.2`) using multiple techniques.

### ✅ Methods

1. **Rule-Based Classifier from Association Rules**
   ("If review includes `Bars` + `Long text` → likely helpful")

2. **Logistic Regression**
   (Structured features only)

3. **Random Forest**

4. **XGBoost**

5. **Multi-Layer Perceptron (MLP)**

6. **KNN**

7. **TF-IDF + Metadata Hybrid**
   (Add review text as sparse features)

8. **Evaluation**

   * Accuracy, F1, Precision/Recall
   * Confusion Matrix, ROC-AUC
   * Rule Coverage vs. ML Performance

---

## ⚙️ Environment Setup

### ✅ Python Version

* Python 3.10+

### ✅ Required Packages

Install dependencies using:

```
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
networkx
xgboost
tqdm
mlxtend
sentence-transformers
```

---

## ▶️ How to Run

### 1. Preprocess Yelp Dataset

```
python3 preprocess.py 
```

### 2. Run Pattern Mining

```
python3 pattern-mining/helpfulness.py
```

### 3. Run Clustering

```
python3 clustering/clustering.py
python3 clustering/hierach-clustering.py
python3 clustering/dbscan.py
```

### 4. Run Classification

```
python3 classification/classification.py
python3 classification/rule-classification.py
python3 classification/mlp.py
```

---

## 📁 Project Structure

```
yelp/
│
├── classification/                          # 🎯 Predicting review helpfulness
│   ├── classification.py                    # ML models: LR, RF, XGBoost
│   ├── mlp.py                               # DL classifier
│   └── rule-classification.py               # Rule-based classifier
│
├── clustering/                              # 📈 User clustering
│   ├── clustering.py                        # K-Means
│   ├── dbscan.py                            # DBSCAN
│   └── hierach-clustering.py                # Hierarchical
│
├── data/                                    # 📂 Processed Yelp subset
│   ├── business.json
│   ├── checkin.json
│   ├── review.json
│   ├── tip.json
│   └── user.json
│
├── pattern-mining/                          # 📊 Association rules
│   ├── helpfulness.py                       # Pattern mining by usefulness
│   └── city-itemset.py                      # City-specific pattern mining
│
├── utils/
│   └── utils.py                             # Helpers for data loading, viz
│
├── preprocess.py                            # 🔄 Dataset preprocessing pipeline
├── requirements.txt                         # 📦 Python dependencies
├── .gitignore                               # 🔒 Ignore data/cache
└── README.md                                # 📘 This file
```

---

## 🧪 Technologies

* Python 3.10
* `pandas`, `numpy`, `scikit-learn`, `mlxtend`
* `matplotlib`, `seaborn`, `plotly`, `networkx`
* `xgboost`, `sentence-transformers`

---

## 🎓 CS412 Concepts Used

| Concept                                    | Where Used |
| ------------------------------------------ | ---------- |
| Data Cleaning & Preprocessing              | Phase 1    |
| Association Rule Mining                    | Phase 2    |
| Constraint-Based Mining                    | Phase 2    |
| Frequent Pattern Summarization             | Phase 2    |
| Clustering (K-Means, DBSCAN, Hierarchical) | Phase 3    |
| Classification & Model Evaluation          | Phase 4    |
| Pattern-Based Classification               | Phase 4    |
| Visual Analytics                           | Phase 2–4  |

---

## 💡 Extensions (Optional)

* Use BERT embeddings for review text
* Add temporal trend analysis (per-user or per-city)
* Create a web dashboard (e.g., with Streamlit)

---

## 📣 Acknowledgments

* Yelp Open Dataset
* Open-source contributors to `mlxtend`, `scikit-learn`, `plotly`

---
