import gradio as gr
import fitz  # PyMuPDF
from groq import Groq
import os

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_input) -> str:
    """Extract text from uploaded PDF - handles both file object and file path"""
    try:
        # Gradio 5+ can pass either file-like object or file path
        if hasattr(pdf_input, 'read'):  # File-like object
            pdf_bytes = pdf_input.read()
        else:  # File path (NamedString or str)
            with open(pdf_input, "rb") as f:
                pdf_bytes = f.read()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def generate_mcqs(text: str, num_questions: int = 10) -> str:
    if not text or len(text.strip()) < 100:
        return "Error: PDF text is too short or empty."

    prompt = f"""You are an expert educator. Create **exactly {num_questions}** high-quality multiple choice questions (MCQs) from the provided text.

Rules:
- Each question must have exactly 4 options (A, B, C, D)
- Only ONE correct answer per question
- Questions should test understanding (not just rote recall)
- Distractors should be plausible
- Vary difficulty and question types
- Return ONLY the MCQs in this exact format. No extra text.

Question 1: [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

Question 2: ...

Text:
{text[:14000]}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise MCQ generator."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=4096,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating MCQs: {str(e)}"

def pdf_to_mcqs(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file."
    
    text = extract_text_from_pdf(pdf_file)
    if text.startswith("Error extracting text"):
        return text
    return generate_mcqs(text, 10)

# Gradio Interface
demo = gr.Interface(
    fn=pdf_to_mcqs,
    inputs=gr.File(label="Upload PDF Document", file_types=[".pdf"]),
    outputs=gr.Textbox(label="Generated 10 MCQs", lines=35),
    title="📚 PDF to MCQ Generator (Groq AI)",
    description="Upload any PDF and get 10 high-quality multiple choice questions instantly!",
    flagging_mode="never",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()