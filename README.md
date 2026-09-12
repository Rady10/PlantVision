# 🌿 PlantVision — AI Plant Disease & Agriculture Assistant

PlantVision is a multimodal AI agriculture project that combines a **fine-tuned Qwen3-VL vision-language model** with a **Retrieval-Augmented Generation (RAG)** knowledge base.

The system is designed to analyze plant images for **disease and pest identification**, explain visible symptoms and characteristics, and provide practical agriculture guidance using retrieved knowledge from an agriculture-focused document collection.

It combines two major AI pipelines:

```text
                    PlantVision
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      VLM Fine-Tuning                RAG Pipeline
             │                           │
             ▼                           ▼
       Qwen3-VL 2B                 AgroLLM PDFs
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 Agricultural AI Assistant
```

---

# ✨ Project Highlights

- 🌱 Plant disease and pest identification
- 👁️ Multimodal image + text understanding
- 🧠 Fine-tuned **Qwen3-VL-2B-Instruct**
- 🪶 Parameter-efficient **LoRA / QLoRA**
- ⚡ **Unsloth** optimized training and inference
- 💾 4-bit quantization
- 📚 Agriculture-focused RAG knowledge base
- 🔎 FAISS semantic retrieval
- 🌍 English + Arabic retrieval support
- 💬 Conversational agricultural assistant
- 🚀 Streaming responses with Gradio
- ☁️ Hugging Face model and dataset hosting
- 🔄 Automatic training checkpoint and resume system

---

# 🏗️ Complete System Architecture

```text
                         ┌──────────────────────────┐
                         │      Farmer / User       │
                         │  Text + Optional Image   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       Gradio UI          │
                         │  MultimodalTextbox       │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴──────────────────┐
                    │                                    │
                    ▼                                    ▼
          ┌───────────────────┐                ┌──────────────────┐
          │    User Query     │                │   Plant Image    │
          └─────────┬─────────┘                └────────┬─────────┘
                    │                                   │
                    ▼                                   │
       ┌─────────────────────────┐                       │
       │ Multilingual Embedding  │                       │
       │ MiniLM-L12-v2           │                       │
       └────────────┬────────────┘                       │
                    │                                    │
                    ▼                                    │
       ┌─────────────────────────┐                       │
       │      FAISS Search       │                       │
       │        Top-K = 3        │                       │
       └────────────┬────────────┘                       │
                    │                                    │
                    ▼                                    │
       ┌─────────────────────────┐                       │
       │ Retrieved Agricultural  │                       │
       │ Knowledge Chunks        │                       │
       └────────────┬────────────┘                       │
                    │                                    │
                    └────────────────┬───────────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │ System Prompt + RAG     │
                         │ Context + Chat History  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Fine-tuned Qwen3-VL 2B  │
                         │        4-bit             │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    Streaming Generation │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       Gradio Chat        │
                         └──────────────────────────┘
```

---

# 🧠 Part 1 — Vision-Language Model

## Base Model

The project starts from:

```text
unsloth/Qwen3-VL-2B-Instruct
```

Qwen3-VL is a multimodal vision-language model capable of processing:

- Images
- Natural language
- Multimodal conversations

The project specializes this general-purpose model for plant pathology and entomology.

---

# 🎯 Model Objective

The fine-tuned model receives:

```text
Plant / Leaf Image
        +
Plant pathology instruction
```

and generates:

```text
Disease / Pest Identification
        +
Symptoms
        +
Visual Characteristics
        +
Morphological Features
        +
Detailed Explanation
```

The training instruction is based on:

```text
You are an expert in plant pathology and entomology.
Analyze this image and identify the disease or pest,
then provide a detailed description of symptoms,
characteristics, and morphological features.
```

---

# 📊 Vision Model Dataset

The fine-tuning dataset is:

```text
Rady10/Plant-Diseases-Image-Text-Pairs
```

Hugging Face:

https://huggingface.co/datasets/Rady10/Plant-Diseases-Image-Text-Pairs

The dataset contains image-text pairs where:

- The image represents a plant/disease example.
- The accompanying text provides the target description.

The notebook shuffles the dataset:

```python
dataset.shuffle(seed=42)
```

A training pool of up to:

```text
100,000 samples
```

is selected.

However, the current training workflow processes:

```text
50,000 samples
```

through five sequential 10,000-sample rounds.

Therefore:

> **Actual samples used for fine-tuning: 50,000**

---

# 🧩 Multimodal Data Format

Each dataset sample is converted into a conversational format:

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

This format allows the model to learn the relationship between visual symptoms and textual agricultural descriptions.

---

# 🪶 Fine-Tuning Techniques

## 1. Supervised Fine-Tuning — SFT

Training is performed using:

```text
TRL SFTTrainer
```

The model learns from:

```text
Image + Instruction → Target Response
```

This teaches Qwen3-VL to produce domain-specific plant pathology responses.

---

## 2. LoRA

Instead of updating every model parameter, the project applies **Low-Rank Adaptation (LoRA)**.

Configuration:

| Parameter | Value |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Bias | none |
| Target modules | all-linear |
| Random seed | 3407 |
| RSLoRA | disabled |

The adaptation targets:

- Vision layers
- Language layers
- Attention modules
- MLP modules

---

## 3. QLoRA / 4-bit Quantization

The base model is loaded using:

```python
FastVisionModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)
```

4-bit quantization reduces GPU memory requirements and makes fine-tuning more practical on constrained hardware.

---

## 4. Unsloth

The project uses:

```text
Unsloth
```

for optimized vision-language fine-tuning and inference.

Gradient checkpointing is configured using:

```text
use_gradient_checkpointing="unsloth"
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|---|---:|
| Base model | Qwen3-VL-2B-Instruct |
| Maximum sequence length | 2048 |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | all-linear |
| Epochs / round | 1 |
| Samples / round | 10,000 |
| Training rounds | 5 |
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

### Precision

The notebook automatically chooses:

```text
BF16 → when supported
FP16 → otherwise
```

---

# 🔄 Sequential Training Workflow

Instead of training all 50K samples in one long run:

```text
Round 1 → 10,000 samples
Round 2 → 10,000 samples
Round 3 → 10,000 samples
Round 4 → 10,000 samples
Round 5 → 10,000 samples
--------------------------------
Total   → 50,000 samples
```

Each round continues from the same LoRA adapter.

---

# 💾 Checkpointing & Automatic Resume

After each round, the adapter is uploaded to Hugging Face.

```text
round_1/
round_2/
round_3/
round_4/
round_5/
final_50k/
```

If training is interrupted:

```text
Round 1 ✓
Round 2 ✓
Round 3 ✓
Round 4 ✗
Round 5 ✗
```

the notebook detects the latest completed checkpoint and resumes from:

```text
Round 4
```

This is especially useful for GPU environments with limited session time.

---

# 🧪 Model Evaluation

The notebook performs qualitative evaluation at two stages.

## Before Fine-Tuning

The original Qwen3-VL model is tested on sample plant images:

```text
Plant Image
     ↓
Base Qwen3-VL
     ↓
Generated Diagnosis
     ↓
Compare with Ground Truth
```

## After Fine-Tuning

The final model is tested on five samples from the end of the selected dataset pool.

The generated responses are displayed alongside the ground-truth descriptions.

### Important

The current notebook does **not** calculate formal:

- Accuracy
- Precision
- Recall
- F1
- BLEU
- ROUGE

The evaluation is primarily qualitative.

---

# 📦 Model Outputs

## Training Checkpoints

```text
Rady10/Plant-Disease-Qwen3VL-2B-Checkpoints
```

Contains intermediate training stages:

```text
round_1 → 10K
round_2 → 20K
round_3 → 30K
round_4 → 40K
round_5 → 50K
```

## Final LoRA Adapter

```text
Rady10/Plant-Disease-Qwen3VL-2B-LoRA
```

## Final Merged Model

```text
Rady10/Plant-Disease-Qwen3VL-2B
```

The merged model is the model used by the runtime application.

---

# 📚 Part 2 — Retrieval-Augmented Generation

The second major component is the agricultural knowledge base.

The purpose of RAG is to provide the model with external agricultural information at inference time instead of relying only on knowledge stored in its parameters.

The RAG architecture is:

```text
Agricultural Documents
        ↓
Text Extraction
        ↓
Text Cleaning
        ↓
Chunking
        ↓
Multilingual Embeddings
        ↓
FAISS Vector Index
        ↓
Semantic Retrieval
        ↓
Qwen3-VL
```

---

# 📖 RAG Dataset

The RAG pipeline uses the:

```text
viswambhar/agrollm-data
```

dataset from Kaggle.

The dataset contains:

```text
31 PDF documents
```

organized into four domains:

```text
Agri_life_sciences          5 PDFs
Agricultural_management     8 PDFs
Agriculture_and_forestry    3 PDFs
Agriculture_business       15 PDFs
```

### Domains

1. Agri Life Sciences
2. Agricultural Management
3. Agriculture & Forestry
4. Agriculture Business

---

# 🔄 RAG Data Preparation

## Step 1 — Download PDFs

The Kaggle API downloads and extracts:

```text
viswambhar/agrollm-data
```

---

## Step 2 — PDF Text Extraction

The project uses:

```text
PyMuPDF
```

for PDF processing.

Each PDF is processed page-by-page.

The text is cleaned by:

- Collapsing excessive whitespace
- Removing line-break artifacts
- Removing null characters
- Ignoring mostly empty pages

Documents containing 100 words or fewer are skipped.

---

# ✂️ Step 3 — Text Chunking

Long documents are divided into smaller overlapping chunks.

Configuration:

| Parameter | Value |
|---|---:|
| Chunk size | 300 words |
| Chunk overlap | 50 words |

The overlap helps preserve context between neighboring chunks.

Each chunk retains metadata such as:

```json
{
  "text": "...",
  "source": "document.pdf",
  "domain": "Agricultural Management"
}
```

---

# 🔢 Step 4 — Multilingual Embeddings

Each chunk is converted into a vector using:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The model was selected to support multilingual semantic retrieval, including:

```text
English
Arabic
```

Embeddings are normalized:

```python
normalize_embeddings=True
```

---

# 🔎 Step 5 — FAISS Vector Search

The project uses:

```text
FAISS
```

with:

```text
IndexFlatIP
```

where `IP` means Inner Product.

Because the embeddings are normalized, inner product behaves like cosine similarity.

At runtime:

```text
Query
 ↓
Embedding
 ↓
FAISS
 ↓
Top 3 chunks
```

Only results with similarity:

```text
> 0.3
```

are included in the generated context.

---

# 💾 RAG Index Files

The generated index consists of:

```text
agro_rag_index/
├── agro.index
├── chunks.json
└── config.json
```

### `agro.index`

Stores the FAISS vector index.

### `chunks.json`

Stores:

- Chunk text
- Source document
- Domain metadata

### `config.json`

Stores configuration such as:

- Embedding model
- Embedding dimension
- Number of chunks
- Chunk size
- Chunk overlap
- Top-K
- Number of PDFs
- Domains

---

# ☁️ Hugging Face RAG Repository

The RAG index is hosted on Hugging Face.

The supplied notebook uses:

```text
Rady10/Plant-Disease-AgroRAG-Index
```

while the runtime `app.py` references:

```text
Rady10/Agriculture-Rag-Data-Index
```

### ⚠️ Important

These repository names should be synchronized before deployment so that the application downloads the same index produced by the notebook.

---

# 🔍 Runtime RAG Workflow

When a user asks an agriculture question:

```text
User Question
      │
      ▼
Multilingual Embedding Model
      │
      ▼
Query Vector
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top 3 Chunks
      │
      ▼
Similarity > 0.3
      │
      ▼
Relevant Agriculture Context
      │
      ▼
System Prompt
      │
      ▼
Qwen3-VL
      │
      ▼
Final Answer
```

---

# 🌍 Arabic & English Support

The multilingual embedding model allows semantic retrieval for Arabic and English.

The application also instructs the generation model to answer in the same language as the user.

```text
English Query
     ↓
English Retrieval
     ↓
English Answer
```

```text
Arabic Query
     ↓
Arabic Retrieval
     ↓
Arabic Answer
```

The RAG notebook tests Arabic retrieval with:

```text
ما هي أمراض القمح؟
```

---

# 🧠 Part 3 — Combining Fine-Tuning + RAG

The most important part of the project is that the two approaches solve different problems.

## Fine-Tuning

Fine-tuning teaches the model:

```text
How to understand plant images
        +
How to respond as a plant pathology assistant
```

## RAG

RAG provides:

```text
External agricultural knowledge
        +
Relevant reference context
```

Together:

```text
                User
                  │
          Image + Question
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   Plant Image         Text Question
        │                   │
        │              Embedding
        │                   │
        │                FAISS
        │                   │
        │             Top-K Context
        │                   │
        └─────────┬─────────┘
                  ▼
        System Prompt + Context
                  │
                  ▼
        Fine-tuned Qwen3-VL
                  │
                  ▼
        Agricultural Answer
```

This allows the system to combine **visual understanding** with **retrieved domain knowledge**.

---

# 💬 Prompt Engineering

The runtime system prompt defines the assistant as an expert in:

```text
Plant pathology
Pest identification
Precision agriculture
```

The assistant is instructed to:

- Answer in the same language as the user.
- Avoid unnecessary jargon.
- Provide practical field-ready advice.
- Give concise responses for normal questions.
- Provide more detail when explicitly requested.

### Image Analysis

For plant images, the assistant is instructed to:

1. Identify the disease or pest if identifiable.
2. Describe 2–3 visible symptoms.
3. Provide one immediate action.

---

# 🖼️ Image Inference Workflow

```text
Plant Image
     │
     ▼
Gradio MultimodalTextbox
     │
     ▼
PIL Image
     │
     ▼
RGB Conversion
     │
     ▼
Qwen3-VL Processor
     │
     ▼
Fine-tuned Qwen3-VL
     │
     ▼
Disease / Pest Explanation
```

If the user uploads only an image, the application generates an automatic instruction asking the model to identify the disease/pest and describe visible symptoms.

---

# 💬 Conversation Workflow

The application maintains conversation history.

Example:

```text
User:
[Plant Image]

Assistant:
The plant appears to show ...

User:
What should I do next?

Assistant:
You should ...
```

The active image can remain available for follow-up questions without requiring the user to upload it again.

---

# ⚡ Streaming Generation

The application uses:

```text
TextIteratorStreamer
```

from Transformers.

Generation occurs in a separate thread:

```text
Qwen3-VL
    ↓
model.generate()
    ↓
TextIteratorStreamer
    ↓
Generated text
    ↓
Gradio chatbot
```

This provides a progressive response instead of waiting for the entire generation to finish.

---

# 🖥️ User Interface

The application uses:

```text
Gradio
```

with:

```python
gr.Blocks()
gr.MultimodalTextbox()
```

The interface provides:

- Chatbot
- Image upload
- Text input
- Multimodal queries
- Clear conversation
- Streaming responses
- English / Arabic interaction

---

# 🛠️ Complete Technology Stack

## AI / Machine Learning

- Python
- PyTorch
- Qwen3-VL
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Hub
- PEFT
- TRL
- Unsloth

## Fine-Tuning

- Vision-Language Model Fine-Tuning
- Supervised Fine-Tuning (SFT)
- LoRA
- QLoRA
- 4-bit Quantization
- Parameter-Efficient Fine-Tuning

## RAG

- Sentence Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`
- FAISS
- PyMuPDF
- NumPy

## Application

- Gradio
- Pillow
- TextIteratorStreamer

## Data / Hosting

- Kaggle
- Hugging Face Hub

---

# 📊 Project Data Summary

| Component | Details |
|---|---|
| Vision base model | Qwen3-VL-2B-Instruct |
| Fine-tuning dataset | Rady10/Plant-Diseases-Image-Text-Pairs |
| Dataset pool | Up to 100K |
| Actual training samples | 50K |
| Training rounds | 5 |
| Samples / round | 10K |
| Fine-tuning | SFT |
| PEFT | LoRA / QLoRA |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| RAG dataset | viswambhar/agrollm-data |
| RAG documents | 31 PDFs |
| RAG domains | 4 |
| PDF extraction | PyMuPDF |
| Chunk size | 300 words |
| Chunk overlap | 50 words |
| Embedding model | multilingual MiniLM-L12-v2 |
| Vector database | FAISS |
| FAISS index | IndexFlatIP |
| Retrieval Top-K | 3 |
| Similarity threshold | > 0.3 |
| UI | Gradio |

---

# 📁 Suggested Repository Structure

```text
plantvision/
│
├── README.md
├── app.py
│
├── notebooks/
│   ├── plant-disease-vlm-unsloth-qwen3vl.ipynb
│   └── Build_RAG_Index.ipynb
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

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd plantvision
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

# 📦 Install Dependencies

For model training:

```bash
pip install unsloth unsloth_zoo
pip install datasets trl evaluate rouge_score huggingface_hub
```

For the RAG pipeline:

```bash
pip install kaggle pymupdf
pip install sentence-transformers faiss-cpu
```

For the application:

```bash
pip install torch transformers pillow gradio numpy
```

---

# 🔐 Hugging Face Authentication

The training notebook uses:

```python
from huggingface_hub import login

login()
```

A Hugging Face account/token with the appropriate permissions is required to upload checkpoints and model artifacts.

Never commit tokens or API credentials to GitHub.

---

# ▶️ Running the Training

Open:

```text
notebooks/plant-disease-vlm-unsloth-qwen3vl.ipynb
```

Run the notebook cells sequentially.

The notebook:

```text
1. Loads dataset
2. Shuffles dataset
3. Creates training pool
4. Formats multimodal conversations
5. Loads Qwen3-VL
6. Applies LoRA
7. Runs zero-shot evaluation
8. Trains 10K samples
9. Saves checkpoint
10. Uploads checkpoint
11. Repeats until 50K
12. Exports final models
```

---

# ▶️ Running the RAG Index Builder

Open:

```text
notebooks/Build_RAG_Index.ipynb
```

The notebook:

```text
1. Downloads AgroLLM dataset
2. Extracts PDFs
3. Extracts PDF text
4. Cleans text
5. Filters short documents
6. Chunks text
7. Generates embeddings
8. Builds FAISS index
9. Saves index files
10. Uploads the index to Hugging Face
11. Tests English and Arabic retrieval
```

---

# ▶️ Running the Application

Before launching the application, verify that the Hugging Face repository configured in `app.py` contains:

```text
agro.index
chunks.json
config.json
```

Then:

```bash
python app.py
```

The Gradio interface will start.

---

# 🔬 End-to-End Project Workflow

```text
                    DATA PREPARATION
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
 Plant Disease Dataset              AgroLLM PDFs
 Image + Text Pairs                Agriculture Knowledge
            │                             │
            ▼                             ▼
       Shuffle Data                  Extract Text
            │                             │
            ▼                             ▼
       Format Messages               Clean Text
            │                             │
            ▼                             ▼
      Qwen3-VL Base                   Chunk Text
            │                             │
            ▼                             ▼
      4-bit Loading                 Generate Embeddings
            │                             │
            ▼                             ▼
         LoRA                     Build FAISS Index
            │                             │
            ▼                             ▼
      SFT Training                 Upload RAG Index
            │                             │
            ▼                             │
    5 × 10K Training Rounds              │
            │                             │
            ▼                             │
     Final LoRA Model                    │
            │                             │
            └──────────────┬──────────────┘
                           ▼
                    RUNTIME SYSTEM
                           │
                           ▼
                    User Image/Query
                           │
                           ▼
                       Gradio UI
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       Plant Image                  User Question
             │                           │
             │                      Embedding
             │                           │
             │                         FAISS
             │                           │
             │                       Top 3 Chunks
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  Prompt + RAG Context
                           │
                           ▼
                  Fine-tuned Qwen3-VL
                           │
                           ▼
                   Streaming Answer
                           │
                           ▼
                         User
```

---

# 🎯 Why Combine Fine-Tuning and RAG?

The two techniques complement each other.

### Fine-Tuning

Changes the model's learned behavior:

```text
General VLM
     ↓
Plant Pathology VLM
```

It teaches the model how to interpret plant images and generate specialized responses.

### RAG

Adds external knowledge dynamically:

```text
Agricultural Documents
        ↓
Relevant Knowledge
        ↓
Model Context
```

This means the knowledge base can be updated independently from the model.

### Combined

```text
Visual Understanding
        +
Domain-Specific Behavior
        +
External Agricultural Knowledge
        ↓
More useful Agricultural AI Assistant
```

---

# ⚠️ Limitations

The current project should be considered a research/prototype system rather than a definitive agricultural diagnostic tool.

Potential limitations include:

- Dataset quality and label accuracy
- Similar visual symptoms between diseases
- Lighting and image-quality variation
- Hallucination in generated explanations
- Limited quantitative evaluation
- No expert-validated diagnostic benchmark in the supplied notebooks
- RAG repository-name mismatch between notebook and runtime
- No explicit confidence/uncertainty estimation

For important agricultural decisions, predictions should be validated by qualified agricultural or plant pathology professionals.

---

# 🔮 Future Improvements

## Model Evaluation

- Dedicated train/validation/test split
- Disease classification accuracy
- Precision / Recall / F1
- Confusion matrix
- Expert evaluation
- Quantitative image-diagnosis benchmark

## RAG Evaluation

- Recall@K
- Precision@K
- MRR
- Retrieval quality evaluation
- Source citations in generated answers
- Domain/crop metadata filtering

## Dataset Improvements

- Class balancing
- Duplicate detection
- Image quality filtering
- Data augmentation
- More plant species
- More disease and pest categories

## Training Improvements

- Hyperparameter experiments
- LoRA rank experiments
- Learning-rate tuning
- More epochs
- Validation loss tracking
- Early stopping

## Application Improvements

- Confidence scores
- Disease treatment recommendations
- Crop-specific recommendations
- Agricultural expert verification
- FastAPI backend
- Flutter mobile application
- Production web application
- User/session management
- Image history

---

# 🔗 Hugging Face Resources

### Dataset

`Rady10/Plant-Diseases-Image-Text-Pairs`

https://huggingface.co/datasets/Rady10/Plant-Diseases-Image-Text-Pairs

### Base Model

`unsloth/Qwen3-VL-2B-Instruct`

https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct

### Training Checkpoints

`Rady10/Plant-Disease-Qwen3VL-2B-Checkpoints`

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B-Checkpoints

### Final LoRA

`Rady10/Plant-Disease-Qwen3VL-2B-LoRA`

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B-LoRA

### Final Merged Model

`Rady10/Plant-Disease-Qwen3VL-2B`

https://huggingface.co/Rady10/Plant-Disease-Qwen3VL-2B

---

# 👨‍💻 Author

**Rady10**

Hugging Face:

https://huggingface.co/Rady10

---

# 📄 License

Check the licenses of the underlying dataset, base model, fine-tuned model, and any third-party resources before redistribution or commercial use.

---

# ⭐ Project

## PlantVision

> **Vision-Language AI + RAG for Plant Disease, Pest Diagnosis & Agricultural Assistance**

