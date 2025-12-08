#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.kaggle.com/code/rishitjakharia/research-paper-summarizer?scriptVersionId=187246482" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

# # RAG Based LLM Application Made using Llama-3-8b-gguf
# ---
# This Application should help provide summary of a reseearch paper when given the name of the research paper

# ## Tech Stack Used
# ----
# 1) arXiv
# 
# 2) LlamaCpp
# 
# 3) sentence-transformers/all-MiniLM-L6-v2

# ## Pages
# ---
# 1) Downloading Relevant libraries/models
# 
# 2) Importing Files
# 
# 3) File Path
# 
# 4) RAG

# ## Downloading relevant libraries/models

# In[1]:




# In[2]:


from langchain_huggingface import HuggingFaceEmbeddings
EMB = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2', 
    model_kwargs=
    {
        'device': 'cuda'
    }
)

# In[3]:


!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.77 -U --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# ## Importing Files

# In[4]:


from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from arxiv import Search
import pandas as pd
import numpy as np
import os
import json
import warnings
import logging
import time
import psutil
import os

# ## File Path

# In[5]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ## RAG

# In[6]:


warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# In[7]:


class QAbot():
    def __init__(self, prompt, Embeddings):
        self.EMBEDDINGS = Embeddings
        self.custom_prompt_template = prompt
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = self.load_llm()
        self.refer = ""
        
    # setting prompt wrapper
    def set_custom_prompt(self, context: str, question: str):
        """
        Create a prompt using the provided context and question.
        """
        prompt_template = PromptTemplate(template=self.custom_prompt_template, input_variables=['context', 'question'])
        prompt = prompt_template.format(context=context, question=question)
        logger.info("The prompt was created.")
        print("Prompt Created")
        return prompt
    
    # Fetching papers from arxiv.org
    def fetch_arxiv_data(self, query):
        search = Search(
            query=query,
            max_results=1, 
        )
        papers = list(search.results())
        sorted_papers = sorted(papers, key=lambda paper: paper.published, reverse=True)  # Sort by publication date
        logger.info("The data retrieved was created.")
        print("Papers Retrieved Created")
        return sorted_papers
    
    # Loading the model
    def load_llm(self):
        """
        Loading the LLM using Llama Cpp.
        """
        llm = LlamaCpp(
            model_path="/kaggle/input/meta-llama/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
            callback_manager = self.callback_manager,
            n_ctx=3072,
            max_tokens = 4096,
            n_gpu_layers=-1,
            verbose=True,
            flash_attn=True,
            chat_format = "llama-3"
        )
        logger.info("LLM model loaded.")
        print("LLM Model Loaded")
        return llm

    # loading the qa_bot wrapper
    def qa_bot(self, question: str):
        """
        Manually perform the search, create the prompt, and get the response from the LLM.
        """
        arxiv_data = self.fetch_arxiv_data(question)
        context = ""
        for papers in arxiv_data:
            context += "**" + papers.title + "**\n"
            context += papers.summary + "\n\n"
        
        prompt = self.set_custom_prompt(context, question)
        print(f"Prompt\n{prompt}\n-----------------------------------")
        return self.llm.invoke(prompt)
    
    def get_reference(self):
        return self.refer

# In[8]:


prompt = """
    based on the following metadata from arxiv.org
    {context}
    provide a text summary for the research papers.
    in format
    ## Title
    - content in points
"""

qa_instance = QAbot(prompt, EMB)

# In[9]:


def main():
    start_time = time.time()
    ans = []
#     question = input("Enter your question below\n")
    question = "NARUTO"
    final = ""
    l = []
    for chunk in qa_instance.qa_bot(question):
        l.append(time.time())
        final += str(chunk)
    _o = qa_instance.get_reference()
    ans.append([question, final, _o])

# In[10]:


if __name__ == "__main__":
    main()

# In[1]:


from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

# Homepage route
def index():
    return render_template('index.html')

# API for summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    title = request.json.get('title')
    # Example mock LLM API call (replace with your model endpoint)
    response = {
        "summary": f"Summary for '{title}': This research covers key insights on RAG models."
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# In[4]:


jupyter nbconvert --to script SummariserUsingLlama.ipynb


# In[ ]:



