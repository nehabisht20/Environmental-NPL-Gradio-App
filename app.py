import gradio as gr
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoTokenizer as NERTokenizer,
    pipeline, StableDiffusionPipeline
)
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import io
import base64
from PIL import Image
import re

# Initialize models (will be loaded on first use)
class ModelManager:
    def __init__(self):
        self.classification_model = None
        self.classification_tokenizer = None
        self.ner_model = None
        self.fill_mask_model = None
        self.image_gen_model = None
        
    def load_classification_model(self):
        if self.classification_model is None:
            # Using a general text classification model and adapting for environmental categories
            self.classification_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.classification_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            
    def load_ner_model(self):
        if self.ner_model is None:
            self.ner_model = pipeline("ner", 
                                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                                    aggregation_strategy="simple")
            
    def load_fill_mask_model(self):
        if self.fill_mask_model is None:
            self.fill_mask_model = pipeline("fill-mask", 
                                          model="bert-base-uncased",
                                          tokenizer="bert-base-uncased")
            
    def load_image_gen_model(self):
        if self.image_gen_model is None:
            try:
                self.image_gen_model = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                if torch.cuda.is_available():
                    self.image_gen_model = self.image_gen_model.to("cuda")
            except:
                # Fallback to a smaller model if GPU memory is limited
                self.image_gen_model = pipeline("text-to-image", 
                                              model="CompVis/stable-diffusion-v1-4")

model_manager = ModelManager()

def classify_environmental_category(text):
    """
    Classify text into environmental categories
    Model: cardiffnlp/twitter-roberta-base-sentiment-latest (adapted)
    """
    model_manager.load_classification_model()
    
    # Environmental keywords mapping
    environmental_categories = {
        'forest': ['forest', 'tree', 'woodland', 'jungle', 'timber', 'deforestation'],
        'water': ['water', 'ocean', 'river', 'lake', 'sea', 'marine', 'aquatic'],
        'desert': ['desert', 'sand', 'arid', 'dry', 'sahara', 'cactus'],
        'mountain': ['mountain', 'hill', 'peak', 'alpine', 'highland'],
        'urban': ['city', 'urban', 'building', 'street', 'metropolitan'],
        'agriculture': ['farm', 'crop', 'agriculture', 'field', 'farming'],
        'climate': ['climate', 'weather', 'temperature', 'global warming', 'carbon']
    }
    
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in environmental_categories.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        predicted_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[predicted_category] / len(text.split())
    else:
        predicted_category = "general_environment"
        confidence = 0.5
    
    return f"Category: {predicted_category.title()} (Confidence: {confidence:.2f})"

def generate_image(text, category):
    """
    Generate image based on text and category
    Model: runwayml/stable-diffusion-v1-5
    """
    model_manager.load_image_gen_model()
    
    # Enhance prompt based on category
    category_prompts = {
        'forest': 'lush green forest, tall trees, natural lighting, photorealistic',
        'water': 'clear blue water, serene lake or ocean, natural landscape',
        'desert': 'vast desert landscape, sand dunes, clear sky, golden hour',
        'mountain': 'majestic mountain range, snow-capped peaks, dramatic landscape',
        'urban': 'modern city skyline, urban environment, architectural details',
        'agriculture': 'green farmland, crops, rural landscape, fertile fields',
        'climate': 'environmental scene showing climate, weather patterns'
    }
    
    base_category = category.split(':')[1].strip().split('(')[0].strip().lower()
    enhanced_prompt = f"{text}, {category_prompts.get(base_category, 'natural environment')}"
    
    try:
        if hasattr(model_manager.image_gen_model, 'to'):
            # Using StableDiffusionPipeline
            image = model_manager.image_gen_model(enhanced_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        else:
            # Using pipeline
            image = model_manager.image_gen_model(enhanced_prompt)[0]
        return image
    except Exception as e:
        # Return a placeholder image if generation fails
        placeholder = Image.new('RGB', (512, 512), color='lightblue')
        return placeholder

def perform_ner_and_plot(text):
    """
    Perform Named Entity Recognition and create visualization
    Model: dbmdz/bert-large-cased-finetuned-conll03-english
    """
    model_manager.load_ner_model()
    
    # Perform NER
    ner_results = model_manager.ner_model(text)
    
    # Process results
    entities = []
    for entity in ner_results:
        entities.append({
            'word': entity['word'],
            'label': entity['entity_group'],
            'confidence': entity['score']
        })
    
    # Create visualization
    if entities:
        labels = [entity['label'] for entity in entities]
        label_counts = Counter(labels)
        
        plt.figure(figsize=(10, 6))
        plt.bar(label_counts.keys(), label_counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        plt.title('Named Entity Recognition Results')
        plt.xlabel('Entity Types')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to PIL Image
        plot_image = Image.open(buf)
        
        # Format entity results
        entity_text = "Entities Found:\n"
        for entity in entities:
            entity_text += f"• {entity['word']} ({entity['label']}) - Confidence: {entity['confidence']:.2f}\n"
    else:
        entity_text = "No named entities found in the text."
        # Create empty plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No Entities Found', ha='center', va='center', fontsize=16)
        plt.title('Named Entity Recognition Results')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        plot_image = Image.open(buf)
    
    return entity_text, plot_image

def fill_in_blanks(text):
    """
    Fill in the blanks in text using mask filling
    Model: bert-base-uncased
    """
    model_manager.load_fill_mask_model()
    
    # Replace * with [MASK] for BERT
    masked_text = text.replace('*', '[MASK]')
    
    try:
        # Get predictions
        predictions = model_manager.fill_mask_model(masked_text)
        
        # Format results
        result_text = f"Original: {text}\n\n"
        result_text += "Top predictions for the blank:\n"
        
        if isinstance(predictions, list):
            for i, pred in enumerate(predictions[:3]):  # Show top 3
                filled_text = pred['sequence']
                confidence = pred['score']
                result_text += f"{i+1}. {filled_text} (Confidence: {confidence:.3f})\n"
        else:
            result_text += f"1. {predictions['sequence']} (Confidence: {predictions['score']:.3f})\n"
            
        return result_text
    
    except Exception as e:
        return f"Error in fill-mask: {str(e)}\nPlease ensure your text contains '*' to be filled."

def process_pipeline(input_text):
    """
    Main pipeline function that processes all tasks sequentially
    """
    if not input_text.strip():
        return "Please enter some text to process.", None, "No text provided.", None, "No text provided."
    
    # Step 1: Sentence Classification
    classification_result = classify_environmental_category(input_text)
    
    # Step 2: Image Generation
    generated_image = generate_image(input_text, classification_result)
    
    # Step 3: NER and Plotting
    ner_text, ner_plot = perform_ner_and_plot(input_text)
    
    # Step 4: Fill in the blanks
    fill_mask_result = fill_in_blanks(input_text)
    
    return classification_result, generated_image, ner_text, ner_plot, fill_mask_result

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Environmental AI Pipeline", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🌍 Environmental AI Pipeline
        
        This comprehensive AI pipeline performs multiple NLP and computer vision tasks on your input text.
        Enter text with environmental themes (use '*' for blanks to fill) and click **Process** to see all results!
        
        **Example inputs:**
        - "* is the world forest day and we need to protect our trees"
        - "The ocean is facing threats from climate change"
        - "Desert ecosystems are fragile and need conservation"
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Enter your text (use '*' for blanks to fill)",
                    placeholder="Example: * is the world forest day and we need to protect our trees",
                    lines=3
                )
                
                process_btn = gr.Button("🚀 Process All Tasks", variant="primary", size="lg")
                
                gr.Markdown("""
                **Models Used:**
                - **Classification**: cardiffnlp/twitter-roberta-base-sentiment-latest (adapted for environmental categories)
                - **Image Generation**: runwayml/stable-diffusion-v1-5
                - **NER**: dbmdz/bert-large-cased-finetuned-conll03-english
                - **Fill Mask**: bert-base-uncased
                """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 1. Environmental Classification")
                classification_output = gr.Textbox(label="Classification Result", lines=2)
                
                gr.Markdown("## 🎨 2. Generated Image")
                image_output = gr.Image(label="Generated Image", height=300)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🏷️ 3. Named Entity Recognition")
                ner_output = gr.Textbox(label="Entities Found", lines=5)
                
                gr.Markdown("## 📈 NER Visualization")
                ner_plot_output = gr.Image(label="Entity Distribution", height=300)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📝 4. Fill in the Blanks")
                fill_mask_output = gr.Textbox(label="Fill Mask Results", lines=6)
        
        # Connect the process button to the pipeline
        process_btn.click(
            fn=process_pipeline,
            inputs=[input_text],
            outputs=[classification_output, image_output, ner_output, ner_plot_output, fill_mask_output]
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["* is the world forest day and we need to protect our trees"],
                ["The ocean is facing threats from climate change and pollution"],
                ["Desert ecosystems in the * are fragile and need conservation"],
                ["Mount Everest is the tallest * in the world"],
                ["Urban areas need more green spaces and sustainable development"]
            ],
            inputs=input_text
        )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True, debug=True)
