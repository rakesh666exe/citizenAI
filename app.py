
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🧠 Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✨ Response generation
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# 🌆 City analysis feature
def city_analysis(city_name):
    prompt = f"""
    Provide a detailed analysis of {city_name} including:
    1. Crime Index and safety statistics
    2. Accident rates and traffic safety information
    3. Overall safety assessment
    
    City: {city_name}
    Analysis:
    """
    return generate_response(prompt, max_length=1000)

# 🏛️ Citizen interaction feature
def citizen_interaction(query):
    prompt = f"""
    As a government assistant, provide accurate and helpful information about the following citizen query related to public services, government policies, or civic issues:

    Query: {query}
    Response:
    """
    return generate_response(prompt, max_length=1000)


# 🎨 Gradio UI Design
custom_css = """
body {background: linear-gradient(135deg, #1e3c72, #2a5298);}
.gradio-container {font-family: 'Poppins', sans-serif;}
h1, h2, h3, h4 {color: white !important; text-align: center;}
button {font-weight: bold;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    gr.HTML("<h1>🏙️ Citizen AI — City & Government Intelligence</h1>")
    gr.Markdown(
        "### 🤖 Analyze Cities or Ask Government-related Questions\n"
        "Get real-time AI-powered insights into safety, transport, and civic services."
    )

    with gr.Tab("🌆 City Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                city_input = gr.Textbox(
                    label="Enter City Name 🏙️",
                    placeholder="e.g., New York, London, Mumbai...",
                    lines=1
                )
                analyze_btn = gr.Button("🔍 Analyze City", variant="primary")
            with gr.Column(scale=2):
                city_output = gr.Textbox(
                    label="City Analysis Report",
                    lines=15,
                    show_copy_button=True
                )

        analyze_btn.click(city_analysis, inputs=city_input, outputs=city_output)

    with gr.Tab("🏛️ Citizen Services"):
        with gr.Row():
            with gr.Column(scale=1):
                citizen_query = gr.Textbox(
                    label="Your Query 💬",
                    placeholder="Ask about public services, government policies, or civic issues...",
                    lines=4
                )
                query_btn = gr.Button("📨 Get Information", variant="primary")
            with gr.Column(scale=2):
                citizen_output = gr.Textbox(
                    label="Government Assistant Response",
                    lines=15,
                    show_copy_button=True
                )

        query_btn.click(citizen_interaction, inputs=citizen_query, outputs=citizen_output)

    gr.Markdown(
        "<div style='text-align:center; color:white; font-size:14px; margin-top:20px;'>"
        "🧠 Powered by IBM Granite & Gradio | Built with ❤️ for Citizen Insights"
        "</div>"
    )

app.launch(share=True)