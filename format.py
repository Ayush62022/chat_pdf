# html_formatting.py

def generate_custom_html_content():
    """
    Generates HTML content with CSS for custom styling in the Streamlit app.
    """
    style = """
    <style>
        .custom-header {
            color: #ffffff;
            background-color: #3333ff;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    """
    
    html_content = f"""
    {style}
    <div class="custom-header">Ask a question about the document</div>
    """

    return html_content
