import sys
import pandas as pd
import gradio as gr
import joblib

labels = ["History", "Language Arts", "Math", "Science", "Social Studies"]


def analyze_text(subtitle):
    model = joblib.load("models/tf_idf_classifier.joblib")
    print(model)
    prediction = model.predict([subtitle])[0]
    print(prediction)
    return {label: prediction[i] for i, label in enumerate(labels)}


def analyze_file(file_path):
    model = joblib.load("models/tf_idf_classifier.joblib")
    print(model)
    df = pd.read_csv(file_path)
    print(df)

    prediction = model.predict(df["text"])
    print(prediction)
    return prediction  # {label: prediction[i] for i, label in enumerate(labels)}


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
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
    neutral_hue="gray",
    secondary_hue=gr.themes.Color(
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
    font=["Segoe UI", "ui-sans-serif", "system-ui", "sans-serif"],
).set(body_background_fill="*code_background_fill", embed_radius="*radius_sm")


def create_webapp():
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("Classhook: A web app to classify videos based on their subtitles")
        # Tab for answer_questions func
        with gr.Tab("One video to classify"):
            with gr.Row():
                with gr.Column():
                    txt_input = gr.Textbox(
                        label="Video Subtitle",
                        type="text",
                        placeholder="Enter video subtitle here...",
                        lines=3,
                    )
                    txt_button = gr.Button("Get Classification")
                with gr.Column():
                    labels_confidence = gr.Label(
                        num_top_classes=5, label="Predicted labels", show_label=False
                    )

        txt_button.click(analyze_text, inputs=[txt_input], outputs=labels_confidence)

        with gr.Tab("Files"):
            file_input = gr.File(label="context")
            file_button = gr.Button("Get Classification")
            file_output = gr.Textbox(
                label="Answer",
                type="text",
                placeholder="Answer will be displayed here...",
            )

        def file_handler(file):
            file_path = file.name
            return analyze_file(file_path)

        file_button.click(file_handler, inputs=[file_input], outputs=file_output)

    demo.launch()


if __name__ == "__main__":
    create_webapp()
