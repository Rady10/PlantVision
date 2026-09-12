# 🌿 PlantVision: Plant Disease & Pest Diagnosis VLM

A vision-language model fine-tuning project for **plant disease and pest identification** using **Qwen3-VL-2B-Instruct**, **Unsloth**, and **LoRA/QLoRA**.

The model is trained to analyze a plant image and generate a detailed response identifying the disease or pest while describing symptoms, characteristics, and morphological features.

---

## 🚀 Project Overview

**PlantVision-Qwen** fine-tunes a small multimodal vision-language model for plant pathology and entomology tasks.

The project uses:

- **Base VLM:** `unsloth/Qwen3-VL-2B-Instruct`
- **Dataset:** `Rady10/Plant-Diseases-Image-Text-Pairs`
- **Training approach:** Supervised Fine-Tuning (SFT)
- **Parameter-efficient fine-tuning:** LoRA / QLoRA
- **Quantization:** 4-bit model loading
- **Training framework:** Unsloth + TRL
- **Training data:** 50,000 samples
- **Training strategy:** 5 sequential rounds × 10,000 samples
- **Checkpointing:** Hugging Face Hub after every round
- **Final outputs:** LoRA adapter and merged 16-bit model

---

## 🔗 Important Links

### Dataset

**Plant-Diseases-Image-Text-Pairs**

https://huggingface.co/datasets/Rady10/Plant-Diseases-Image-Text-Pairs

### Base Model

**Qwen3-VL-2B-Instruct**

https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct

### Training Checkpoints

**Plant-Disease-Qwen3VL-2B-Checkpoints**

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B-Checkpoints

### Final LoRA Model

**Plant-Disease-Qwen3VL-2B-LoRA**

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B-LoRA

### Final Merged Model

**Plant-Disease-Qwen3VL-2B**

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B

---

## 🎯 Objective

The objective is to adapt a general-purpose vision-language model to a specialized agricultural domain.

Given a plant image, the model receives the following instruction:

> You are an expert in plant pathology and entomology. Analyze this image and identify the disease or pest, then provide a detailed description of symptoms, characteristics, and morphological features.

The model then generates a text response based on the visual information in the image.

### Input

- Plant/leaf image
- Domain-specific instruction

### Output

- Disease or pest identification
- Symptoms
- Visual characteristics
- Morphological features
- Detailed textual explanation

---

---
## Flow Diagram
<img width="1312" height="1199" alt="ChatGPT Image Sep 12, 2026, 01_58_43 PM" src="https://github.com/user-attachments/assets/7acad7b9-dd75-4026-8f20-99957e09d8ba" />

---

## 🧠 Techniques Used

### 1. Vision-Language Model Fine-Tuning

The project fine-tunes a multimodal model capable of processing both:

- Images
- Natural language

This allows the model to learn relationships between visual plant symptoms and textual disease/pest descriptions.

---

### 2. QLoRA / 4-bit Fine-Tuning

The base model is loaded using 4-bit quantization:

```python
FastVisionModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)
```

This reduces GPU memory requirements while allowing parameter-efficient fine-tuning.

---

### 3. LoRA

Low-Rank Adaptation is applied instead of fully fine-tuning every model parameter.

Configuration used:

| Parameter | Value |
|---|---:|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Bias | none |
| Target modules | all-linear |
| Random seed | 3407 |
| RSLoRA | disabled |

The project fine-tunes:

- Vision layers
- Language layers
- Attention modules
- MLP modules

This makes the adaptation more domain-specific while keeping the number of trainable parameters much smaller than full fine-tuning.

---

### 4. Supervised Fine-Tuning (SFT)

The model is trained using `TRL`'s `SFTTrainer`.

Each training example follows a conversational multimodal format:

```text
User:
[Image] + Plant pathology instruction

Assistant:
Disease/pest description
```

This teaches the model to generate the desired domain-specific response from an image.

---

### 5. Chunked Training

Instead of processing the entire training set in one training run, the project divides training into sequential chunks.

```text
Round 1 → 10,000 samples
Round 2 → 10,000 samples
Round 3 → 10,000 samples
Round 4 → 10,000 samples
Round 5 → 10,000 samples
--------------------------------
Total   → 50,000 samples
```

Each round continues training the same LoRA model.

---

### 6. Hugging Face Checkpointing

After every training round, the adapter is saved and uploaded to the Hugging Face Hub.

```text
round_1/
round_2/
round_3/
round_4/
round_5/
final_50k/
```

This provides fault tolerance and makes long training jobs easier to resume.

---

### 7. Automatic Resume

When the notebook starts, it checks the checkpoint repository for completed rounds.

If previous checkpoints exist, the latest adapter is downloaded and loaded.

Example:

```text
Round 1 ✓
Round 2 ✓
Round 3 ✓
Round 4 ✗
Round 5 ✗

Resume → Round 4
```

This avoids repeating completed training rounds after an interruption.

---

## 📊 Dataset

The project uses:

**Rady10/Plant-Diseases-Image-Text-Pairs**

Dataset:

https://huggingface.co/datasets/Rady10/Plant-Diseases-Image-Text-Pairs

The dataset provides image-text pairs where the image represents a plant/disease example and the accompanying text provides the target description.

The notebook shuffles the dataset using:

```python
dataset.shuffle(seed=42)
```

It then creates a training pool of up to **100,000 samples**:

```python
TOTAL_SAMPLES = 100000
```

However, the current training loop processes only the first **50,000 samples**, using five 10K rounds.

Therefore:

> **Actual samples used for fine-tuning: 50,000**

The remaining samples in the selected 100K pool are not used by the five-round training loop and can be useful for future evaluation or additional training.

---

## 🔄 Complete Workflow

```text
                    ┌──────────────────────┐
                    │ Plant Disease Dataset│
                    │ Image + Text Pairs    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Shuffle Dataset      │
                    │ seed = 42             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Select Training Pool │
                    │ up to 100K samples   │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Convert to Multimodal Messages  │
              │ Image + Instruction → Response  │
              └────────────────┬────────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Qwen3-VL-2B-Instruct     │
                 │ 4-bit Quantized Loading  │
                 └────────────┬─────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Apply LoRA / QLoRA │
                    │ r=16, α=16         │
                    └─────────┬──────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ Zero-Shot Inference        │
                │ Before Fine-Tuning         │
                └─────────────┬──────────────┘
                              │
                              ▼
             ┌──────────────────────────────────┐
             │ Sequential SFT Training          │
             │                                  │
             │ Round 1 → 10K                    │
             │ Round 2 → 10K                    │
             │ Round 3 → 10K                    │
             │ Round 4 → 10K                    │
             │ Round 5 → 10K                    │
             └────────────────┬─────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Save LoRA Adapter  │
                    │ After Every Round  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Hugging Face Hub   │
                    │ Checkpoint Upload  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Final 50K Adapter  │
                    └─────────┬──────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ Post-Fine-Tuning Inference │
                └─────────────┬──────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Publish Final Models    │
                 │                         │
                 │ LoRA Adapter            │
                 │ Merged 16-bit Model     │
                 └─────────────────────────┘
```

---

## ⚙️ Training Configuration

The main training configuration is:

| Parameter | Value |
|---|---:|
| Base model | Qwen3-VL-2B-Instruct |
| Maximum sequence length | 2048 |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | all-linear |
| Epochs per round | 1 |
| Samples per round | 10,000 |
| Number of rounds | 5 |
| Total training samples | 50,000 |
| Batch size / device | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Scheduler | Linear |
| Warmup steps | 5 |
| Weight decay | 0.01 |
| Optimizer | AdamW 8-bit |
| Seed | 3407 |
| Max new tokens (pre-training test) | 200 |
| Max new tokens (post-training test) | 300 |

### Precision

The notebook automatically chooses:

- **BF16** when supported
- Otherwise **FP16**

```python
fp16 = not torch.cuda.is_bf16_supported()
bf16 = torch.cuda.is_bf16_supported()
```

---

## 🧩 Data Formatting

Raw dataset samples are converted into a multimodal conversation.

Conceptually:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this plant image..."
        },
        {
          "type": "image"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Disease/pest description..."
        }
      ]
    }
  ]
}
```

This format is compatible with the Unsloth vision data collator and TRL SFT training pipeline.

---

## 🧪 Evaluation Strategy

The notebook performs qualitative evaluation at two stages.

### Before Fine-Tuning — Zero-Shot Test

The original Qwen3-VL model is tested on sample plant images before training.

The generated response is compared visually with the dataset ground truth.

```text
Plant Image
     ↓
Base Qwen3-VL
     ↓
Generated Diagnosis
     ↓
Compare with Ground Truth
```

### After Fine-Tuning

The fine-tuned model is tested on samples from the end of the selected dataset pool.

The notebook uses the final five samples:

```python
test_indices = list(range(len(dataset)-5, len(dataset)))
```

The generated answers are displayed alongside the ground-truth descriptions.

### Important Evaluation Note

The notebook does **not** currently calculate a formal accuracy, F1, BLEU, ROUGE, or other quantitative evaluation score.

Although `evaluate` and `rouge_score` are installed, the shown workflow uses qualitative visual/text comparison instead.

For a production/research benchmark, a dedicated held-out test set and quantitative metrics should be added.

---

## 📦 Model Outputs

### 1. Intermediate Checkpoints

Each training stage creates an adapter checkpoint:

```text
round_1 → 10K samples
round_2 → 20K samples
round_3 → 30K samples
round_4 → 40K samples
round_5 → 50K samples
```

These are stored in:

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B-Checkpoints

---

### 2. Final LoRA Adapter

The final LoRA adapter is published as:

**Plant-Disease-Qwen3VL-2B-LoRA**

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B-LoRA

This is useful when you want to keep the base model separate and load the learned LoRA weights on top.

---

### 3. Final Merged Model

The project also exports a merged 16-bit model:

**Plant-Disease-Qwen3VL-2B**

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B

This provides a standalone model artifact without requiring a separate LoRA adapter during inference.

---

## 🛠️ Tech Stack

### AI / Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Hub
- PEFT
- TRL
- Unsloth
- Qwen3-VL

### Fine-Tuning

- Vision-Language Model Fine-Tuning
- Supervised Fine-Tuning (SFT)
- LoRA
- QLoRA
- 4-bit Quantization
- Parameter-Efficient Fine-Tuning

### Training Optimization

- Gradient Accumulation
- 8-bit AdamW
- BF16 / FP16
- Linear Learning Rate Scheduler
- Warmup
- Checkpointing
- Automatic Resume

---

## 📁 Suggested Repository Structure

```text
plantvision-qwen/
│
├── README.md
├── notebooks/
│   └── plant-disease-vlm-unsloth-qwen3vl.ipynb
│
├── checkpoints/
│   ├── round_1/
│   ├── round_2/
│   ├── round_3/
│   ├── round_4/
│   └── round_5/
│
├── models/
│   ├── lora/
│   └── merged/
│
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd plantvision-qwen
```

### 2. Install dependencies

The notebook uses:

```bash
pip install unsloth unsloth_zoo
pip install datasets trl evaluate rouge_score huggingface_hub
```

### 3. Authenticate with Hugging Face

The notebook uses:

```python
from huggingface_hub import login

login()
```

You need a Hugging Face account/token with permission to create and upload the required model/checkpoint repositories.

### 4. Run the notebook

Open:

```text
notebooks/plant-disease-vlm-unsloth-qwen3vl.ipynb
```

Run the cells sequentially.

If training is interrupted, re-running the notebook allows the checkpoint logic to detect completed rounds and continue from the latest available checkpoint.

---

## 🔁 Fault-Tolerant Training

One of the key design decisions is the checkpoint-per-round workflow.

Instead of:

```text
50K samples
    ↓
one long training run
    ↓
failure
    ↓
restart everything ❌
```

the project uses:

```text
10K → checkpoint ✓
10K → checkpoint ✓
10K → checkpoint ✓
10K → checkpoint ✓
10K → checkpoint ✓
```

If a failure occurs after Round 3:

```text
Round 1 ✓
Round 2 ✓
Round 3 ✓
Round 4 ✗
Round 5 ✗
```

the notebook detects the completed rounds and resumes from the latest checkpoint.

This is especially useful for GPU environments with limited session time.

---

## 💡 Why LoRA + Unsloth?

Full fine-tuning of a vision-language model can require substantial GPU memory and compute.

This project instead combines:

```text
Qwen3-VL
   +
4-bit Quantization
   +
LoRA
   +
Unsloth
   +
Gradient Accumulation
```

The goal is to make domain adaptation more practical on constrained GPU hardware while retaining the capabilities of the original multimodal model.

---

## 🔬 Potential Improvements

The current notebook provides a strong fine-tuning pipeline, but the project can be extended with:

### Evaluation

- Dedicated train/validation/test split
- Disease classification accuracy
- Precision / Recall / F1
- ROUGE / BLEU for generated descriptions
- Human expert evaluation
- Confusion matrix by disease category

### Dataset

- Class balancing
- Duplicate detection
- Image quality filtering
- Data augmentation
- More diverse plant species
- More diverse disease and pest categories

### Training

- Hyperparameter experiments
- Different LoRA ranks
- Learning-rate tuning
- More training epochs
- Validation loss tracking
- Early stopping

### Deployment

- FastAPI inference API
- Flutter mobile application
- Web-based plant diagnosis interface
- Image upload + diagnosis workflow
- Confidence/uncertainty estimation
- Agricultural recommendation layer

---

## ⚠️ Limitations

This model should be treated as a research/prototype system rather than a definitive agricultural diagnostic tool.

Potential limitations include:

- Dataset quality and label accuracy
- Similar visual symptoms across different diseases
- Variation in lighting and image quality
- Limited evaluation in the current notebook
- Possible hallucination in generated explanations
- No expert-validated diagnostic benchmark included in the notebook

For real-world agricultural decisions, predictions should be validated by qualified plant pathology/agriculture professionals.

---

## 📈 Project Highlights

- 🌱 Specialized plant disease/pest VLM
- 👁️ Multimodal image + text understanding
- 🧠 Qwen3-VL-2B foundation model
- ⚡ Unsloth optimized fine-tuning
- 🪶 LoRA parameter-efficient adaptation
- 💾 4-bit quantized training
- 🔄 Automatic checkpoint/resume workflow
- ☁️ Hugging Face Hub integration
- 📦 50K-sample sequential training pipeline
- 🚀 Final LoRA and merged model exports

---

## 👨‍💻 Author

**Rady10**

Hugging Face:

https://huggingface.co/Rady10

---

## 📄 License

The appropriate license for the final model and dataset should be checked on their respective Hugging Face repositories before redistribution or commercial use.

---

## ⭐ Project Name

### **PlantVision**

**Tagline:**  
> *Vision-Language AI for Plant Disease & Pest Diagnosis*

