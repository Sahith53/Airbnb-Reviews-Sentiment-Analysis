Research Paper Summarizer (Llama + arXiv)
=========================================

Overview
--------
This project is a Kaggle notebook export that summarizes an arXiv paper using a
local Llama model (GGUF) and LangChain. The script currently mixes notebook
cells, a CLI demo, and a Flask demo in one file, so it needs a small cleanup
and local configuration to run on Windows.

Files
-----
- SummariserUsingLlama.py - main script (Kaggle notebook export)

Requirements
------------
- Windows 10/11
- Python 3.10+ recommended
- A local GGUF model file (for example, Llama 3 8B Instruct)

Setup (PowerShell)
------------------
1) Create and activate a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install dependencies
   pip install --upgrade pip
   pip install langchain langchain-core langchain-community langchain-huggingface
   pip install arxiv pandas numpy psutil flask requests
   pip install llama-cpp-python

If you have an NVIDIA GPU and CUDA already configured, you can rebuild
llama-cpp-python with CUDA support. Refer to the llama-cpp-python docs for the
latest Windows CUDA build instructions.

Local Configuration
-------------------
Open `SummariserUsingLlama.py` and update the following:

1) Remove notebook-only lines (they will crash in a normal .py run)
   - The line starting with `!CMAKE_ARGS=... pip install ...`
   - The line `jupyter nbconvert --to script SummariserUsingLlama.ipynb`

2) Update model path to your local GGUF
   Example:
   model_path="C:\\models\\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"

3) If you do NOT have a GPU, change embeddings device to CPU
   - In `HuggingFaceEmbeddings`, set `device` to `cpu`

4) Choose one entry point (CLI or Flask) and disable the other
   - The script currently has two `if __name__ == "__main__":` blocks.
   - Comment out the one you are not using.

Run: CLI Demo
-------------
1) In `main()`, set your query:
   question = "your paper title or topic"

2) Run the script:
   python SummariserUsingLlama.py

Run: Flask API (mocked response)
--------------------------------
The Flask section currently returns a mock summary and the `index()` route
is missing a decorator. To use the API:

1) Add a route decorator above `index()`:
   @app.route("/")
   def index():
       return render_template("index.html")

2) Create `templates/index.html` if you want a homepage, or remove the
   homepage route entirely.

3) Comment out the CLI `if __name__ == "__main__": main()` block.

4) Run:
   python SummariserUsingLlama.py

Then POST JSON to:
   http://localhost:5000/summarize
   body: {"title": "your paper title"}

Notes
-----
- This code was written for Kaggle and needs the local updates above.
- The arXiv search is limited to the most recent matching result.
- The Llama model invocation is local and can be slow without GPU support.

If you want, I can clean the script into two separate files (CLI + Flask) and
make it runnable out of the box.
