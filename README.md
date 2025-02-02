# Text Similarity Model Evaluation using TOPSIS

This project evaluates multiple pre-trained text similarity models by applying the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** decision-making technique. The evaluation considers various similarity metrics to identify the best model for text similarity tasks.

## 🚀 Project Overview

The project compares several state-of-the-art text embedding models to compute the similarity between sentence pairs using different metrics. The results are then ranked using the TOPSIS method to find the best-performing model.

### Key Components
- **Sentence Transformer Models:** Five pre-trained models for text similarity.
- **Evaluation Metrics:** Cosine similarity, Euclidean distance, and Manhattan distance.
- **Decision-Making Technique:** TOPSIS to rank models based on multiple criteria.

## 📊 Models Evaluated
- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- `paraphrase-MiniLM-L6-v2`
- `distiluse-base-multilingual-cased-v1`
- `bert-base-nli-mean-tokens`

## 📁 Dataset
The dataset used for evaluation is the **STS-B (Semantic Textual Similarity Benchmark) multilingual dataset** from Hugging Face.

```python
from datasets import load_dataset

# Load the English test split from the dataset
dataset = load_dataset("stsb_multi_mt", name="en", split="test")
text_pairs = list(zip(dataset["sentence1"], dataset["sentence2"]))[:200]
```

## ⚙️ Project Workflow

### Step 1: Compute Similarity Scores
For each sentence pair and each model, compute the following similarity metrics:
- **Cosine Similarity:** Measures angular similarity between sentence embeddings.
- **Euclidean Distance:** Measures straight-line distance between sentence embeddings (negated for ranking in TOPSIS).
- **Manhattan Distance:** Measures grid-like distance between embeddings (negated for ranking in TOPSIS).

### Step 2: Apply TOPSIS
Normalize the computed metrics, assign weights, and compute distances from the ideal best and worst solutions to rank the models.

### Step 3: Visualize Results
The final rankings are visualized using a bar plot to show the performance of each model.

## 💻 Installation and Usage

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `sentence-transformers`, `datasets`, `scikit-learn`

### Installation
```bash
pip install numpy pandas matplotlib sentence-transformers datasets scikit-learn
```

### Running the Project
Copy the project script to `text_similarity_topsis.py` and execute it as follows:

```bash
python text_similarity_topsis.py
```

## 📈 Results

After applying TOPSIS, the models are ranked based on their similarity scores. Below is a sample result:

| Model                          | TOPSIS Score | Rank |
|---------------------------------|--------------|------|
| distiluse-base-multilingual-cased-v1 | 0.985380         | 3    |
| all-MiniLM-L6-v2               | 0.920748         | 4    |
| all-mpnet-base-v2              | 0.894152         | 1    |
| paraphrase-MiniLM-L6-v2        | 0.563682         | 2    |
| bert-base-nli-mean-tokens      | 0.102838         | 5    |

## 📊 Visualization
A bar plot is generated to visualize the model rankings.
## 📊 Visualization
Below is the bar plot generated to visualize the model rankings:

![Model Rankings Visualization](topsis_visualization.png)


