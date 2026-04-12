# Fine-Tuning VGG16 for Dogs vs Cats Classification

Workshop submission: **Fine-Tuning a Neural Network** — adapting a pre-trained VGG16 image classifier (originally trained on ImageNet) to tell dogs from cats with a small custom dataset.
# Student Name/Team Members
Prajesh Bhatt  
KevinKumar Patel
Muthuraj Jayakumar

## What's in this repo

| File | Purpose |
|---|---|
| `05D_fine_tuning_vgg16.ipynb` | The full lecture-style notebook: feature extraction, data augmentation, fine-tuning, comparison, reflection, and a bonus experiment. |
| `requirements.txt` | Pinned Python dependencies for reproducibility. |
| `.gitignore` | Excludes `data/`, `models/`, virtualenvs, and caches. |
| `README.md` | This file. |

## Learning objectives

1. Use a pre-trained convolutional base (VGG16) as a frozen feature extractor.
2. Improve generalization with in-model data augmentation.
3. Fine-tune the last few convolutional layers at a tiny learning rate and compare against the feature-extraction baselines.
4. Answer the three reflection questions in the notebook's challenge.
5. Experiment with freeze depth (bonus).

## How to run

```bash
# 1. Create an isolated environment
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset (see "Dataset" section below)
#    The notebook expects: ./data/kaggle_dogs_vs_cats_small/{train,validation,test}/{cat,dog}/*.jpg

# 4. Launch Jupyter and open the notebook
jupyter lab 05D_fine_tuning_vgg16.ipynb
```

From **Run All**, Kernel → Restart & Run All. Outputs are saved in the notebook so the grader can see the results without re-executing every cell.

## Dataset

The notebook uses a small subset of the Kaggle **Dogs vs Cats** dataset (2000 training + 1000 validation + 1000 test images). It is **not** checked into git. To reproduce:

1. Download from <https://www.kaggle.com/c/dogs-vs-cats/data>.
2. Extract to `./data/kaggle_dogs_vs_cats_small/` with the structure:
   ```
   data/kaggle_dogs_vs_cats_small/
   ├── train/      {cat, dog}/
   ├── validation/ {cat, dog}/
   └── test/       {cat, dog}/
   ```

## Reproducibility notes

- **Python**: 3.12
- **Seeds**: `tf.keras.utils.set_random_seed(42)` is set at the top of the notebook.
- **GPU vs CPU**: The data pipeline uses `.cache().prefetch(AUTOTUNE)` and `jit_compile=True` (with CPU fallback). With a CUDA-visible GPU, full end-to-end training takes roughly 30–60 minutes. On CPU it can take 4+ hours; in that case, load the saved `.keras` files and skip straight to the "Challenge Solution" section.
- **Models**: trained weights are saved to `./models/*.keras` and are gitignored.
- **Batch size**: set to 64 in the notebook. Drop to 32 if you hit GPU out-of-memory errors.

## Credits

Notebook structure adapted from François Chollet's *Deep Learning with Python* (2nd edition) chapter 8 examples, with additional pedagogical commentary, GPU-friendly pipeline upgrades, and the full challenge solution added for this workshop.
