import numpy as np
import gradio as gr
from afterSave import savedPDFWorker
from PDFtest import newPDFWorker
import os

css = """
.feedback textarea {font-size: 24px !important}
"""
PDFfolderName = ""
uploadPDFWorker = newPDFWorker()
PDFWorker = savedPDFWorker()

def upload_file(file):
    PDFfolderName = os.path.basename(file.name).rstrip(".pdf")
    uploadPDFWorker.setFolder(PDFfolderName)
    uploadPDFWorker.embedPDF(file.name)
    print("Done")

def pickFile(name):
    PDFWorker.setFolder(name)
    return "D:\\Documents\\NotesHelper\\" + name

def get_response(message, history):
    result = PDFWorker.get_AI_response(message)
    result_String = ""
    result_String += result["result"]
    result_String += "\nSources:\n"
    sources = result["source_documents"]
    for doc in sources:
        result_String += "Page: "+ str(doc.metadata['page']) + "\n"
        #print("Page Content: ", doc.page_content)
    return result_String

with gr.Blocks(css=css, title="Study Pal") as demo:
    with gr.Accordion("Study Pal!"):
        gr.Markdown("Welcome to Study Pal!\nThis is a tool that will help you study by answering questions you have about your notes, books, or any text in a PDF.\nTry it out!")
    with gr.Tab("Ask"):
        gr.Interface(fn=pickFile, inputs="text", outputs="text")
        chat = gr.ChatInterface(get_response)
    with gr.Tab("Upload"):
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to Upload a File. Your default folder is: D:\\Documents\\NotesHelper", file_types=["pdf"])
        upload_button.upload(upload_file, upload_button, file_output)

demo.launch()