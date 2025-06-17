from llama_parse import LlamaParse
import os
import nest_asyncio
from dotenv import load_dotenv

# Enable nested async loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_API_KEY")

# Initialize the parser
documents = LlamaParse(
    result_type="text",
    user_prompt="""
    This is a very complex PDF file with many tables. Please extract all table content as accurately and in detail as possible.
    """
).load_data("gen_ai.pdf")

# Concatenate all extracted text
all_text = "\n\n".join([doc.text for doc in documents])

# Save the output as a markdown file
md_filename = "parsed_case_2.md"
with open(md_filename, "w", encoding="utf-8") as md_file:
    md_file.write(all_text)

print(f"Markdown saved at: {md_filename}")
