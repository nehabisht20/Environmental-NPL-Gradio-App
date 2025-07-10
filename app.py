import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification
import torch
import matplotlib.pyplot as plt
import networkx as nx
import tempfile
import logging

logging.basicConfig(level=logging.INFO)

# 1. Sentence Classification (environment-related categories)
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# 2. Image Generation via Hugging Face pipeline (text-to-image)
text_to_image = pipeline("text-to-image", model="CompVis/stable-diffusion-v1-4")

# 3. NER pipeline
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# 4. Fill-mask pipeline
fill_mask_pipeline = pipeline("fill-mask", model="bert-base-uncased")


def process_all(input_text):
    result_classification = "Not processed"
    result_image = None
    result_ner_text = "Not processed"
    result_ner_graph = None
    result_mask = "Not processed"

    # Sentence Classification
    try:
        cls = classifier(input_text)
        result_classification = f"Label: {cls[0]['label']} (Model: bhadresh-savani/distilbert-base-uncased-emotion)"
    except Exception as e:
        result_classification = f"Error: {str(e)}"

    # Image Generation
    try:
        # The text-to-image pipeline might require a different input format.
        # This is a common issue after library updates.
        # Check the documentation for the specific model for the correct input.
        image = text_to_image(input_text)[0]['image']
        img_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(img_temp.name)
        result_image = img_temp.name
    except Exception as e:
        result_image = None
        logging.error(f"Image generation failed: {str(e)}. This might be due to an incorrect input format for the pipeline after a library update. Please check the documentation for the text-to-image model ('CompVis/stable-diffusion-v1-4') to confirm the expected input type.")
        result_image = f"Image generation failed: {str(e)}. Check documentation for input format."


    # NER
    try:
        ner_results = ner_pipeline(input_text)
        entities = [ent['word'] for ent in ner_results]
        graph = nx.Graph()
        for i, ent in enumerate(entities):
            graph.add_node(ent)
            if i > 0:
                graph.add_edge(entities[i - 1], ent)

        result_ner_text = f"{[f'{e['entity_group']}: {e['word']}' for e in ner_results]} (Model: dslim/bert-base-NER)"

        fig, ax = plt.subplots()
        nx.draw(graph, with_labels=True, node_color='lightgreen', edge_color='gray')
        graph_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(graph_temp.name)
        plt.close()
        result_ner_graph = graph_temp.name
    except Exception as e:
        result_ner_text = f"NER error: {str(e)}"
        result_ner_graph = None

    # Fill-in-the-blank
    try:
        if "_" in input_text or "[MASK]" in input_text:
            masked_input = input_text.replace("_", "[MASK]")
            fill = fill_mask_pipeline(masked_input)
            result_mask = f"{fill[0]['sequence']} (Model: bert-base-uncased)"
        else:
            result_mask = "No blank (_) found."
    except Exception as e:
        result_mask = f"Mask filling error: {str(e)}"

    return result_classification, result_image, result_ner_text, result_ner_graph, result_mask


# Gradio UI
iface = gr.Interface(
    fn=process_all,
    inputs=gr.Textbox(label="Enter a sentence (use '_' for blank):", lines=4, placeholder="Example: _ is celebrated as World Environment Day"),
    outputs=[
        gr.Text(label="1. Sentence Classification"),
        gr.Image(label="2. Generated Image"),
        gr.Text(label="3. Named Entities (NER)"),
        gr.Image(label="3. NER Graph"),
        gr.Text(label="4. Fill in the Blank"),
    ],
    title="🌱 Environmental AI Toolkit",
    description="Input a sentence with an optional blank (_), then press Submit. This app:\n"
                "- Classifies the topic\n"
                "- Generates a related image\n"
                "- Detects named entities and plots them\n"
                "- Fills in the blank\n\n"
                "**Hugging Face Models Used:**\n"
                "- Sentence Classification: `bhadresh-savani/distilbert-base-uncased-emotion`\n"
                "- Text-to-Image: `CompVis/stable-diffusion-v1-4`\n"
                "- NER: `dslim/bert-base-NER`\n"
                "- Fill-in-the-blank: `bert-base-uncased`"
)

iface.launch()
