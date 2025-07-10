import gradio as gr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForTokenClassification
import torch
import matplotlib.pyplot as plt
import networkx as nx
import tempfile

# 1. Sentence Classification model
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# 2. Image Generation model (BLIP caption model for simple gen)
image_gen_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
image_gen_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# 3. NER model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# 4. Fill-mask model
fill_mask_pipeline = pipeline("fill-mask", model="bert-base-uncased")

def process_all(input_text):
    outputs = {}

    # Step 1: Classification
    classification = classifier(input_text)
    classified_label = classification[0]['label']
    outputs['Classification'] = f"Label: {classified_label} (Model: bhadresh-savani/distilbert-base-uncased-emotion)"

    # Step 2: Image Generation
    inputs = image_gen_processor(input_text, return_tensors="pt")
    out = image_gen_model.generate(**inputs)
    caption = image_gen_processor.decode(out[0], skip_special_tokens=True)

    # Create placeholder image with caption text
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, caption, wrap=True, ha='center', va='center', fontsize=12)
    ax.axis('off')
    temp_img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.savefig(temp_img_path)
    plt.close()

    outputs['Generated Image'] = temp_img_path

    # Step 3: NER + Graph
    ner_results = ner_pipeline(input_text)
    entities = [ent['word'] for ent in ner_results]
    G = nx.Graph()
    for i, ent in enumerate(entities):
        G.add_node(ent)
        if i > 0:
            G.add_edge(entities[i - 1], ent)

    graph_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.figure(figsize=(6, 4))
    nx.draw(G, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.savefig(graph_path)
    plt.close()
    outputs['NER Graph'] = graph_path
    outputs['NER Entities'] = f"{[e['entity_group'] + ': ' + e['word'] for e in ner_results]} (Model: dslim/bert-base-NER)"

    # Step 4: Fill Mask
    if '[MASK]' in input_text or '_' in input_text:
        masked = input_text.replace("_", "[MASK]")
        fill_results = fill_mask_pipeline(masked)
        top_fill = fill_results[0]['sequence']
        outputs['Fill Mask'] = f"{top_fill} (Model: bert-base-uncased)"
    else:
        outputs['Fill Mask'] = "No mask token ('_' or '[MASK]') found."

    return outputs['Classification'], temp_img_path, outputs['NER Entities'], graph_path, outputs['Fill Mask']


# Gradio UI
iface = gr.Interface(
    fn=process_all,
    inputs=gr.Textbox(label="Enter a sentence (use '_' for blank):"),
    outputs=[
        gr.Text(label="1. Sentence Classification"),
        gr.Image(label="2. Generated Image"),
        gr.Text(label="3. Named Entities (NER)"),
        gr.Image(label="3. NER Graph"),
        gr.Text(label="4. Fill in the Blank"),
    ],
    title="Environmental NLP Pipeline 🌍",
    description="Click the button to run all steps sequentially. Models used:\n"
                "- Sentence Classification: `bhadresh-savani/distilbert-base-uncased-emotion`\n"
                "- Image Captioning: `Salesforce/blip-image-captioning-base`\n"
                "- Named Entity Recognition: `dslim/bert-base-NER`\n"
                "- Fill in the Blank: `bert-base-uncased`"
)

iface.launch()
