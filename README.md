# 📧 Spam Detection — Gaussian & Bernoulli Naive Bayes

A machine learning project implementing and comparing **Gaussian** and **Bernoulli Naive Bayes** classifiers for SMS spam detection, with text preprocessing, vectorization, and visual performance analysis.

---

## 📌 Project Overview

This project applies two variants of the Naive Bayes algorithm to the classic SMS Spam Collection dataset. The goal is to classify messages as **spam** or **ham (not spam)** and compare how well each model performs on binary text data. The project covers the full ML pipeline: preprocessing → vectorization → training → evaluation → visualization.

This was a group assignment focused on understanding the **mathematical foundations** of Naive Bayes and its real-world applicability in NLP tasks.

---

## 🧠 Algorithms Implemented

### Gaussian Naive Bayes
Assumes features follow a **normal (Gaussian) distribution**. Estimates the mean (μ) and variance (σ²) of each feature per class:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

> Less ideal for binary text features but included for comparison.

### Bernoulli Naive Bayes
Designed for **binary/boolean features** — perfect for bag-of-words text classification. Models each feature as a Bernoulli trial (word present = 1, absent = 0):

$$P(x_i \mid y) = P(i \mid y)^{x_i} \cdot (1 - P(i \mid y))^{(1-x_i)}$$

> Well-suited for spam detection where word presence/absence matters most.

Both models apply **Bayes' Theorem** at prediction time:

$$P(y \mid x) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Gaussian Naive Bayes | ~87% |
| **Bernoulli Naive Bayes** | **~98%** ✅ |

**Bernoulli NB significantly outperforms Gaussian NB** on this task — as expected, since binary word-presence features are exactly what Bernoulli NB is designed for.

Visualizations generated:
- Confusion matrices for both models (heatmaps)
- Side-by-side accuracy comparison bar chart

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading and preprocessing |
| Scikit-learn | `GaussianNB`, `BernoulliNB`, `CountVectorizer`, metrics |
| NLTK | Stopword removal |
| Matplotlib & Seaborn | Confusion matrix heatmaps and bar charts |
| Jupyter Notebook | Interactive development environment |

---

## 📁 Repository Structure

```
spam-detection-naive-bayes/
│
├── Gaussian_Bernoulli_Naive_Bayes.ipynb   # Full ML pipeline notebook
├── spam.csv                               # SMS Spam Collection dataset
├── visuals/
│   ├── confusion_matrix_gaussian.png      # Generated at runtime
│   ├── confusion_matrix_bernoulli.png     # Generated at runtime
│   └── accuracy_comparison.png           # Generated at runtime
└── README.md                             # Project documentation
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/AmaanJilani1/spam-detection-naive-bayes.git
cd spam-detection-naive-bayes
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

**3. Download NLTK stopwords** (first run only)
```python
import nltk
nltk.download('stopwords')
```

**4. Launch the notebook**
```bash
jupyter notebook Gaussian_Bernoulli_Naive_Bayes.ipynb
```

> The `visuals/` folder will be created automatically when the notebook runs.

---

## 📚 Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 SMS messages
- **Class distribution:** 4,825 ham (86.6%) · 747 spam (13.4%)
- **Split:** 75% training / 25% testing (`random_state=42`)

---

## 🔄 ML Pipeline

```
Raw CSV
   ↓
Label encoding (ham=0, spam=1)
   ↓
Text preprocessing (lowercase → remove punctuation → remove stopwords)
   ↓
Binary vectorization (CountVectorizer with binary=True)
   ↓
Train/Test split (75/25)
   ↓
GaussianNB  ←→  BernoulliNB
   ↓
Accuracy · Confusion Matrix · Classification Report
```

---

## 💡 Key Insights

- **Bernoulli NB is the right tool** for bag-of-words text classification — word presence/absence is inherently binary
- **Gaussian NB underperforms** here because it wrongly assumes continuous feature distributions on sparse binary data
- **Class imbalance** (87% ham) means accuracy alone is misleading — precision and recall for the spam class are the more important metrics
- **Real-world use:** Naive Bayes spam filters power Gmail, Yahoo Mail, and other email clients due to their speed and low memory footprint

---

## 👤 Author

**Amaan Jilani**  
[GitHub](https://github.com/AmaanJilani1) · [LinkedIn](https://www.linkedin.com/in/amaanjilani/)

---

## 📝 Notes

- Completed as part of **DS2001 – Introduction to Data Science**, Assignment #2, FAST-NUCES (Fall 2025)
- Group project — algorithm implementation, mathematical explanation, and visual analysis
