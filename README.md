# Sentiment Analysis & Sarcasm Detection

A comparison of classical machine learning and transformer-based approaches on two English text-classification tasks:

- **Sentiment analysis** on SST-2 (positive vs. negative)
- **Sarcasm detection** on news headlines (sarcastic vs. normal)

The notebook trains and evaluates the models, compares them on accuracy, precision, recall, F1, training time and memory, and ends with an interactive Gradio demo that predicts sentiment and sarcasm for any sentence.

## Datasets

| Task | Dataset | Size used | Labels |
|---|---|---|---|
| Sentiment | SST-2 (GLUE, `nyu-mll/glue`) | 10,000 random samples from the train split | 0 = negative (4,507), 1 = positive (5,493) |
| Sarcasm | News Headlines (`raquiba/Sarcasm_News_Headline`) | 28,619 headlines | 0 = normal (14,985), 1 = sarcastic (13,634) |

Both datasets are fairly balanced. SST-2 texts are short and skewed toward very few words (mostly under 20); sarcasm headlines are tightly concentrated around 8–14 words.

**Splits** (stratified, seed 42): 80% train+val / 20% test, then 10% of the training part held out for validation.

| Task | Train | Val | Test |
|---|---|---|---|
| Sentiment | 7,200 | 800 | 2,000 |
| Sarcasm | 20,605 | 2,290 | 5,724 |

**Preprocessing** is intentionally light: whitespace normalization and removal of URLs and HTML tags. Case and punctuation are kept, since they can carry sarcasm signals.

## Models

| Model | Type | Notes |
|---|---|---|
| TF-IDF + Linear SVM | Classical | 50k features, unigrams + bigrams, sublinear TF |
| TF-IDF + Logistic Regression | Classical | Same features as above |
| Zero-shot NLI | Pre-trained, no fine-tuning | `cross-encoder/nli-distilroberta-base`, evaluated on the first 500 test samples per task |
| DistilBERT / BERT / RoBERTa | Fine-tuned transformers | `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`; 3 epochs, lr 2e-5, batch size 32, max length 128, dynamic padding |

## Results

Metrics are weighted averages on the held-out test set (source: `project1_results.csv`).

### Sentiment (SST-2)

| Model | Accuracy | Precision | Recall | F1 | Train time (s) |
|---|---|---|---|---|---|
| Zero-shot NLI* | 0.8360 | 0.8475 | 0.8360 | **0.8364** | 7.2 |
| TF-IDF + SVM | 0.8210 | 0.8208 | 0.8210 | 0.8206 | 0.2 |
| TF-IDF + LR | 0.8010 | 0.8041 | 0.8010 | 0.7987 | 1.8 |

### Sarcasm (News Headlines)

| Model | Accuracy | Precision | Recall | F1 | Train time (s) |
|---|---|---|---|---|---|
| TF-IDF + SVM | 0.8566 | 0.8566 | 0.8566 | **0.8566** | 0.8 |
| TF-IDF + LR | 0.8442 | 0.8445 | 0.8442 | 0.8442 | 2.1 |
| Zero-shot NLI* | 0.5140 | 0.5101 | 0.5140 | 0.5108 | 5.7 |

\* Zero-shot NLI has no training step; the time shown is inference time on 500 samples, and it was scored on that subset only, so it is not strictly comparable with the other rows.

### Key takeaways

- **Sentiment:** zero-shot NLI scores highest, with no task-specific training. It is a general-purpose model that has learned what "positive" and "negative" mean, so this task suits it well.
- **Sarcasm:** zero-shot NLI performs close to chance (~51%), while a simple TF-IDF + SVM reaches ~86%. Sarcasm in headlines depends on dataset-specific style and vocabulary (the data mixes satirical and real news outlets), which a supervised model picks up and a zero-shot prompt does not.
- **Cost vs. performance:** TF-IDF + SVM is the best trade-off overall: it trains in under a second and is competitive on both tasks.

## Figures

| File | Content |
|---|---|
| `eda_datasets.png` | Label distribution and text-length histograms |
| `f1_comparison.png` | F1 per model, per task |
| `metrics_grouped.png` | Accuracy / precision / recall / F1 side by side |
| `tradeoff.png` | F1 vs. training time (log scale) |
| `confusion_matrices.png` | Transformer confusion matrices |
| `training_curves.png` | Transformer train / validation loss |

## Current limitations

- **The transformer results are not included yet.** The fine-tuning function for DistilBERT, BERT and RoBERTa is defined in the notebook, but the run was done on CPU and the fine-tuning loop was not executed. As a result, `confusion_matrices.png` and `training_curves.png` are empty and `project1_results.csv` contains only the classical and zero-shot models. The sarcasm error-analysis cell (section 9) is skipped for the same reason (no saved RoBERTa model).
- **Demo quality:** because no fine-tuned model is saved, the predictor falls back to TF-IDF + Logistic Regression. It is a weak model for general sentences (for example, it gave a low-confidence "negative" for a clearly positive movie review), so demo predictions should be treated as illustrative.
- **Memory metric is unreliable.** Memory is measured as the change in process RSS, which is noisy on CPU (one value is even negative) and is 0 for zero-shot. Do not draw conclusions from it. The bubble sizes in `tradeoff.png` depend on it.
- **Only 10,000 of the SST-2 training sentences are used**, and results come from a single run with one seed, so small differences (for example SVM vs. LR) may not be significant.
- Both datasets are English only, although the demo interface supports French and English labels.

## Getting started

### Requirements

```bash
pip install transformers datasets scikit-learn accelerate evaluate psutil \
            matplotlib seaborn pandas numpy sentencepiece gradio
```

A GPU is strongly recommended for fine-tuning the transformers (the notebook enables fp16 automatically when CUDA is available). The classical baselines and zero-shot run fine on CPU.

### Run

1. Open `sentiment-sarcasm-nlp.ipynb` in Jupyter, Colab or Kaggle.
2. Adjust the `CFG` dictionary at the top if needed (sample size, epochs, batch size, learning rate, zero-shot sample count, `run_transformers`).
3. Run the cells in order. To get the transformer results, make sure the fine-tuning step (section 6) is actually executed for each model and task, e.g.:

   ```python
   for task in TASKS:
       for short_name, hf_name in CFG["models"].items():
           fine_tune_transformer(hf_name, short_name, task)
   ```

4. Results are saved to `project1_results.csv` and the plots to PNG files in the working directory.

### Interactive demo

The last cells launch a Gradio app (`demo.launch(share=True)`) that takes a sentence and returns:

- the sentiment prediction with a confidence score,
- the sarcasm prediction with a confidence score,
- a short combined interpretation (e.g. "positive on the surface, but sarcastic").

It loads fine-tuned RoBERTa models from `./outputs/RoBERTa_<task>` if they exist, otherwise it falls back to TF-IDF + LR.

## Repository structure

```
.
├── sentiment-sarcasm-nlp.ipynb   # full pipeline: data, training, evaluation, demo
├── project1_results.csv          # metrics for all evaluated models
├── eda_datasets.png
├── f1_comparison.png
├── metrics_grouped.png
├── tradeoff.png
├── confusion_matrices.png
└── training_curves.png
```

## Possible next steps

- Run the DistilBERT / BERT / RoBERTa fine-tuning and add them to the results table and plots.
- Evaluate zero-shot on the full test set (or fine-tune the NLI model) for a fair comparison.
- Use a proper memory measurement (peak GPU memory, model size) and repeat runs with several seeds.
- Test on out-of-domain sarcasm (tweets, reviews) to check whether the headline-trained model generalizes.