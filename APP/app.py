import os
import gradio as gr
from huggingface_hub import login
from transformers import pipeline

hf_token = os.environ.get("HF_Token")
if hf_token:
    login(token=hf_token)
    print("Login HuggingFace berhasil!")
else:
    print("WARNING: HF_Token tidak ditemukan!")

MODEL_NAME = "Ipuldev14/indobert-sentiment-shopee"
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1
)
print("Model siap!")

EMOJI = {"Positif": "[+]", "Negatif": "[-]", "Netral": "[~]"}

def analyze_sentiment(text, history):
    if not text.strip():
        return history, ""
    result     = classifier(text)[0]
    label      = result["label"]
    confidence = result["score"] * 100
    emoji      = EMOJI[label]
    response = (
        "Sentimen: " + emoji + " " + label
        + " (" + "{:.1f}".format(confidence) + "%)\n\n"
        + "Ulasan: \"" + text + "\"\n\n"
        + "Model mengklasifikasikan ulasan ini sebagai " + label
        + " dengan keyakinan " + "{:.1f}".format(confidence) + "%."
    )
    history = history or []
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})
    return history, ""

def clear_chat():
    return [], ""

with gr.Blocks(title="Analisis Sentimen Shopee") as demo:
    gr.Markdown("""
    # Analisis Sentimen Ulasan Shopee
    **Model:** IndoBERT fine-tuned untuk klasifikasi sentimen ulasan Bahasa Indonesia.
    """)
    chatbot = gr.Chatbot(
        label="Riwayat Analisis",
        height=400,
    )
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Contoh: Produk bagus, pengiriman cepat!",
            label="Masukkan ulasan produk",
            scale=4, lines=2
        )
        with gr.Column(scale=1):
            send_btn  = gr.Button("Analisis", variant="primary")
            clear_btn = gr.Button("Hapus")
    gr.Examples(
        examples=[
            ["Aplikasi ini sangat bagus dan mudah digunakan!"],
            ["Pengiriman lambat banget, barang datang rusak. Kecewa!"],
            ["Biasa saja, tidak terlalu istimewa"],
        ],
        inputs=msg, label="Contoh Ulasan"
    )
    send_btn.click(analyze_sentiment, [msg, chatbot], [chatbot, msg])
    msg.submit(analyze_sentiment,     [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch()