import sys
import pandas as pd
import gradio as gr
import joblib
from youtube_transcript_api import YouTubeTranscriptApi
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import torch

import joblib

model_name = "benjaminrio/bert-finetuned-classhook-16"  # "benjaminrio/bert-finetuned-classhook-8"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding=True, truncation=True, max_length=512
)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = -1
tokenizer_kwargs = {
    "padding": True,
    "truncation": True,
    "max_length": 512,
}
if torch.cuda.is_available():
    print("Using GPU: ")
    print(torch.cuda.get_device_name(0))
    device = 0
pipeline = TextClassificationPipeline(
    model=model, tokenizer=tokenizer, top_k=None, device=device
)

labels = [
    "Science",
    "Social Studies",
    "History",
    "Math",
    "Language Arts",
    "Social-Emotional Learning",
    "Life Skills",
    "Business",
    "Physics",
    "Health",
    "Biology",
    "Communication",
    "Philosophy and Theology",
    "Self-awareness",
    "Psychology",
    "Culture",
]  # ["History", "Language Arts", "Math", "Science", "Social Studies"]

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#e0f2fe",
        c200="#bae6fd",
        c300="#7dd3fc",
        c400="#38bdf8",
        c50="#f0f9ff",
        c500="#27a9e1",
        c600="#0284c7",
        c700="#0369a1",
        c800="#075985",
        c900="#0c4a6e",
        c950="#0b4165",
    ),
    neutral_hue="gray",
    secondary_hue=gr.themes.Color(
        c100="#ffedd5",
        c200="#fed7aa",
        c300="#fdba74",
        c400="#fb923c",
        c50="#fff7ed",
        c500="#f05a28",
        c600="#ea580c",
        c700="#c2410c",
        c800="#9a3412",
        c900="#7c2d12",
        c950="#6c2e12",
    ),
    font=["Segoe UI", "ui-sans-serif", "system-ui", "sans-serif"],
).set(body_background_fill="*code_background_fill", embed_radius="*radius_sm")


def analyze_text_tfidf(subtitle):
    model = joblib.load("../models/tf_idf_classifier.joblib")
    print(model)
    prediction = model.predict([subtitle])[0]
    print(prediction)
    return {label: prediction[i] for i, label in enumerate(labels)}


def analyze_test_hf(subtitle):
    prediction = pipeline(subtitle, **tokenizer_kwargs)[0]
    print(prediction)
    return {dics["label"]: round(dics["score"], 2) for dics in prediction}


def download_subtitles(text):
    print(text)
    try:
        video_id = text.split("v=")[1]
    except:
        try:
            video_id = text.split("/")[-1]
        except:
            return "URL not valid..."
    try:
        subs = YouTubeTranscriptApi.get_transcripts([video_id], languages=["en"])
    except:
        return "No subtitles found..."
    sentences = [sentence["text"] for sentence in subs[0][video_id]]
    return " ".join(sentences)


def analyze_file(file):
    df = pd.read_csv(file.name, header=None, names=["text"], encoding="utf-8")
    df["subtitles"] = df["text"].apply(download_subtitles)
    df[labels] = df["subtitles"].apply(analyze_test_hf).apply(pd.Series)
    return df


def export_csv(df):
    df.to_csv("hookie_predictions.csv")
    return gr.File(value="hookie_predictions.csv", visible=True)


def create_webapp():
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("<h1 style='text-align: center;'> Hookie  </h1>")
        with gr.Tab("Youtube link"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Textbox(
                        label="Youtube URL",
                        type="text",
                        placeholder="Enter video URL here...",
                        lines=1,
                    )
                    get_subs_button = gr.Button("Get Subtitles")
                with gr.Column():
                    txt_input = gr.Textbox(
                        label="Subtitle",
                        type="text",
                        placeholder="Enter video subtitle here...",
                        lines=1,
                    )
                    txt_button = gr.Button("Get Classification")
                with gr.Column():
                    labels_confidence = gr.Label(
                        num_top_classes=5, label="Predicted labels", show_label=False
                    )
        get_subs_button.click(
            download_subtitles, inputs=[video_input], outputs=[txt_input]
        )
        txt_button.click(analyze_test_hf, inputs=[txt_input], outputs=labels_confidence)

        with gr.Tab("Files"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Youtube URLs file")
                    file_button = gr.Button("Get Classification")

                with gr.Column(scale=4):
                    dataframe = gr.Dataframe(
                        wrap=False,
                        column_widths=["10%", "18%"] + ["4.5%"] * len(labels),
                    )
                    export_button = gr.Button("Export")
                    csv = gr.File(interactive=False, visible=False)
        file_button.click(analyze_file, inputs=file_input, outputs=dataframe)
        export_button.click(export_csv, inputs=dataframe, outputs=csv)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Share option")
    parser.add_argument("--remote", type=bool, help="share", default=False)
    args = parser.parse_args()
    app = create_webapp()
    app.launch(share=args.remote)
