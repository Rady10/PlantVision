import torch
import threading
import json
import numpy as np
import os
import faiss
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from unsloth import FastVisionModel
from transformers import TextIteratorStreamer
from PIL import Image
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_REPO   = "Rady10/Plant-Disease-Qwen3VL-2B"
RAG_REPO     = "Rady10/Agriculture-Rag-Data-Index"
MAX_SEQ_LENGTH = 512
RAG_TOP_K    = 3        # Number of chunks to retrieve per query
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are AgroVision, an expert AI assistant specialized in plant pathology, pest identification, and precision agriculture.

## Core Behavior
- Respond in the SAME language the user writes in (Arabic or English). Never mix languages.
- Be concise and direct. Answer in 2-5 sentences for standard queries.
- Only expand with more detail if the user explicitly asks for a "full analysis", "detailed report", or similar.
- Never repeat the same information twice.
- Never start with phrases like "According to the image..." or "Based on my analysis..." — just answer directly.

## When Analyzing a Plant Image
1. Name the disease or pest (if identifiable).
2. State the 2-3 most visible symptoms from the image.
3. Suggest one immediate action the farmer can take.
Format: plain prose, NOT bullet lists unless the user asks for a list.

## When Answering a Text Question
- Give a practical, field-ready answer a farmer can act on.
- Prefer simple language over scientific jargon unless the user appears to be a researcher.

## Tone
Warm, knowledgeable, and professional — like a trusted agronomist speaking directly to a farmer.\
"""

print(f"CUDA Available: {torch.cuda.is_available()}")

# ── Load RAG Index ────────────────────────────────────────────────────────────
print("⬇️  Downloading RAG index from Hub...")
rag_dir = snapshot_download(repo_id=RAG_REPO, repo_type="dataset", local_dir="./rag_index")

print("📚 Loading FAISS index and chunks...")
faiss_index = faiss.read_index(os.path.join(rag_dir, "agro.index"))
with open(os.path.join(rag_dir, "chunks.json"), "r", encoding="utf-8") as f:
    rag_chunks = json.load(f)

print(f"RAG index loaded: {faiss_index.ntotal:,} vectors, {len(rag_chunks):,} chunks")

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"Embedder ready: {EMBED_MODEL}")

# ── Load Vision Model ─────────────────────────────────────────────────────────
print(f"Loading {MODEL_REPO}...")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_REPO,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    max_seq_length=MAX_SEQ_LENGTH,
)
FastVisionModel.for_inference(model)
print("Model ready!")

# ── RAG Retrieval ─────────────────────────────────────────────────────────────
def retrieve_context(query: str, top_k: int = RAG_TOP_K) -> str:
    """Embed query and retrieve top-k relevant chunks from FAISS index."""
    if not query.strip():
        return ""
    q_embed = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = faiss_index.search(q_embed, k=top_k)
    parts = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist > 0.3:   # Only include sufficiently relevant chunks
            parts.append(rag_chunks[idx]['text'])
    return "\n\n".join(parts)


# ── App State ──────────────────────────────────────────────────────────────────
active_image_state = {"image": None}


# ── Generation ────────────────────────────────────────────────────────────────
def generate_response(user_text: str, past_history: list, image=None):
    """
    Build messages with optional RAG context and generate a streamed response.
    """
    # 1. Retrieve relevant context from RAG
    rag_context = retrieve_context(user_text)

    # 2. Build system message — include RAG context if found
    if rag_context:
        system_text = (
            SYSTEM_PROMPT
            + "\n\nUse the following knowledge to help answer the user's question:\n\n"
            + rag_context
        )
    else:
        system_text = SYSTEM_PROMPT

    # 3. Build formatted messages
    formatted_messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]}
    ]

    for msg in past_history:
        if msg["role"] == "user":
            if isinstance(msg["content"], tuple):
                continue  # Skip image entries in history (saves VRAM)
            content_str = str(msg["content"]).strip()
            if content_str:
                formatted_messages.append({"role": "user",      "content": [{"type": "text", "text": content_str}]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": str(msg["content"])}]})

    # 4. Current user turn
    current_content = []
    if image is not None:
        current_content.append({"type": "image", "image": image})
    current_content.append({"type": "text", "text": user_text})
    formatted_messages.append({"role": "user", "content": current_content})

    # 5. Tokenize & generate
    input_text = tokenizer.apply_chat_template(formatted_messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text


# ── Gradio Interface ───────────────────────────────────────────────────────────
def chat_interface(user_message, history):
    text  = user_message.get("text", "")
    files = user_message.get("files", [])

    if files:
        active_image_state["image"] = Image.open(files[0]).convert("RGB")

    if files and not text.strip():
        text = (
            "Look at this plant image. Identify the disease or pest if present, "
            "and briefly describe the key visible symptoms in 2-4 sentences."
        )

    if not text.strip() and not files:
        yield history, gr.MultimodalTextbox(value=None, interactive=True)
        return

    past_history = history.copy()

    if files:
        history.append({"role": "user", "content": (files[0],)})
    if text.strip():
        history.append({"role": "user", "content": text})

    history.append({"role": "assistant", "content": ""})

    for output in generate_response(text, past_history, image=active_image_state["image"]):
        history[-1]["content"] = output
        yield history, gr.MultimodalTextbox(value=None, interactive=True)


def clear_memory():
    active_image_state["image"] = None
    return [], gr.MultimodalTextbox(value=None, interactive=True)


# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(title="🌿 Plant Disease & Agriculture Assistant") as demo:
    gr.Markdown("## 🌿 Plant Disease & Agriculture Assistant")
    gr.Markdown(
        "Upload a plant image to identify diseases, or ask any agriculture question in **English or Arabic**. "
        "Powered by **RAG** from the AgroLLM knowledge base + a fine-tuned **Qwen3-VL** vision model."
    )

    chatbot = gr.Chatbot(type="messages", height=500)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image"],
        placeholder="Upload a plant image, or ask a question in English or Arabic...",
        show_label=False,
    )

    chat_input.submit(chat_interface, [chat_input, chatbot], [chatbot, chat_input])

    clear_btn = gr.ClearButton([chatbot, chat_input])
    clear_btn.click(clear_memory, None, [chatbot, chat_input])

    gr.Markdown(
        "---\n"
        "💡 **Tips:** Ask casually — *'What's wrong with this plant?'* — or ask for detail — *'Give me a full analysis.'* "
        "You can also ask follow-up questions without re-uploading the image."
    )

if __name__ == "__main__":
    demo.launch()
