# Sycophancy Reduction via Activation Steering
A mechanistic interpretability project demonstrating how to identify and causally steer behavioral representations in language models using Contrastive Activation Addition (CAA).
## 🎯 Project Overview
This project implements a rigorous approach to representation engineering and activation steering for reducing sycophantic behavior in large language models. Rather than fine-tuning model weights, we identify specific activation patterns that encode sycophancy and intervene directly on the model's internal representations at inference time.
### What is Sycophancy?
Sycophancy is when a model tells users what they want to hear rather than providing honest, accurate information. For example, if a user with clear political biases asks a factual question, a sycophantic model will agree with the user's viewpoint regardless of truth.
### Key Research Questions

1. **Where** in the network are sycophantic behaviors represented?
2. **How** can we extract a steering vector that captures this behavior?
3. **Can** we causally intervene to reduce sycophancy without retraining?
4. **Does** the intervention generalize to new prompts?


## 🔬 Methodology
This project follows rigorous mechanistic interpretability practices:
### 1. Contrastive Dataset Construction

- Uses paired examples of sycophantic vs. non-sycophantic responses
- Dataset: nrimsky/CAA Sycophancy Dataset
- Each pair shares the same question but differs in whether the answer agrees with the user's stated bias

### 2. Layer Selection via Validation

- Train/Validation Split: Prevents overfitting to training data
- Layer Sweep: Tests layers 10-20 (middle layers where high-level concepts are encoded)
- Effect Measurement: Quantifies steering effectiveness using held-out validation set

### 3. Steering Vector Extraction

- Method: Contrastive Activation Addition (CAA)
- Formula: v = mean(honest_activations) - mean(sycophantic_activations)
- Aggregator options: Mean difference (default), PCA, Logistic Regression
- Applied to residual stream activations at selected layer(s)

### 4. Causal Validation

- Test multiple multiplier values: [-2, -1, 0, 1, 2]
- Verify bidirectional effects (negative multipliers reduce sycophancy)
- Evaluate on completely held-out test set
- Compare to baseline and random direction controls

## 📊 Results
### Baseline (No Steering):

- Model shows ~60% sycophantic behavior

### After Steering (multiplier = -1.5):

- Sycophancy reduced to ~30-40%
- Model provides more honest, balanced responses
- Effect generalizes to unseen prompts

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sycophancy-steering.git
cd sycophancy-steering

# Install dependencies
pip install steering-vectors torch transformers accelerate datasets scikit-learn matplotlib tqdm
```
### Requirements:

- Python 3.8+
- PyTorch 2.0+
- 13GB RAM
- GPU with 15GB+ VRAM (The standard T4 GPU available with Google Colab (free tier) will be able to support this project)

## 🚀 Quick Start
### Basic Usage

```python
from steering_vectors import train_steering_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load your training data
positive_samples = [...]  # Non-sycophantic responses
negative_samples = [...]  # Sycophantic responses

# Train steering vector
steering_vec = train_steering_vector(
    model,
    tokenizer,
    positive_samples,
    negative_samples,
    layers=[14],  # Optimal layer for Llama-2-7B
    show_progress=True
)

# Apply steering during generation
with steering_vector.apply(model, multiplier=-1.5):
    output = model.generate(
        tokenizer.encode("Your prompt here", return_tensors="pt"),
        max_new_tokens=100
    )
    print(tokenizer.decode(output[0]))
```

### Complete Pipeline

```bash
# 1. Download datasets
python scripts/download_data.py

# 2. Find optimal layer (includes validation)
python scripts/find_best_layer.py --model meta-llama/Llama-2-7b-chat-hf

# 3. Train final steering vector
python scripts/train_steering.py --layer 14 --output steering_vec.pt

# 4. Evaluate on test set
python scripts/evaluate.py --steering steering_vec.pt --multiplier -1.5

# 5. Interactive demo
python scripts/demo.py --steering steering_vec.pt
```

## 🔍 Key Implementation Details

### Layer Selection Methodology

1. Split data: 70% train, 30% validation
2. For each layer in range [10, 20]:
   - Train steering vector on training set
   - Evaluate steering effect on validation set
   - Record effect magnitude
3. Select layer with maximum steering effect

### Preventing Data Leakage

✅ Separate train/validation/test splits
✅ Train steering vector only on training data
✅ Select layer using validation data
✅ Final evaluation on completely held-out test set
✅ Never evaluate on training data

## 🔗 References
### Papers

- Contrastive Activation Addition: [Rimsky et al. 2023](https://arxiv.org/abs/2308.10248)
- Representation Engineering: [Zou et al. 2023](https://arxiv.org/abs/2310.01405)

## Libraries

- [steering-vectors](https://github.com/steering-vectors/steering-vectors): Production implementation of CAA
- [nrimsky/CAA](https://github.com/nrimsky/CAA): Original CAA implementation and datasets

## Tutorials

- [Activation Steering Tutorial](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector)

## 📝 License
- MIT License - feel free to use this for research or education!

## 🙏 Acknowledgments

- Nina Rimsky and team for the CAA methodology and datasets
- Andy Zou for Representation Engineering framework
- Anthropic for research on constitutional AI and honesty
- steering-vectors library maintainers

### Happy steering! 🎯
