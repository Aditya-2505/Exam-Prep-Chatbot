# 📘 Exam Prep Chatbot

An intelligent AI-powered assistant that transforms the way students study by providing **document-specific, multilingual answers** to academic queries. Built on **Retrieval-Augmented Generation (RAG)** and optimized using **OpenVINO**, the chatbot ensures fast and accurate responses with an intuitive **Gradio interface**.

---

## 🚀 Demo

![Exam Prep Chatbot Demo](https://github.com/user-attachments/assets/e778db1f-bfc3-4e88-b060-7ad3c84cb1e3)

---

## ✨ Key Features

- 🔍 **Document-Based Question Answering**: Upload PDFs of your study materials and get context-aware answers.
- 🌐 **Multilingual Support**: Chat in English or Chinese effortlessly.
- 🧠 **LLM Powered with RAG**: Combines semantic retrieval and generation to deliver accurate results.
- ⚡ **Optimized with OpenVINO**: Enhanced inference speed for better responsiveness.
- 🧩 **Modular Backend Support**: Easily switch between different language models via configuration.
- 💬 **Interactive Gradio UI**: Clean, accessible interface with chat history and examples.
- 📓 **Notebook Exploration**: Dive deep into the RAG pipeline via Jupyter Notebook.

---

## 🛠️ Installation Guide

### 🔧 Prerequisites
- Python 3.8+
- pip or conda

### 📦 Option 1: pip + virtualenv
```bash
# Clone repo
git clone https://github.com/username/exam-prep-chatbot.git
cd exam-prep-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🐍 Option 2: Using conda
```bash
git clone https://github.com/username/exam-prep-chatbot.git
cd exam-prep-chatbot

conda create -n exam-chatbot python=3.9
conda activate exam-chatbot

pip install -r requirements.txt
```

---

## 💡 How to Use

### ▶️ Launch the Chatbot
```bash
python rag_ui.py
```
Open `http://127.0.0.1:7860` in your browser to start chatting.

### 🧪 Explore via Notebook
```bash
jupyter notebook llm-rag-langchain.ipynb
```

### 📚 Chatbot Usage Flow
1. **Upload** your study PDFs
2. **Select Language** (English/Chinese)
3. **Ask** questions related to the content
4. **Review** contextual answers
5. **Use Examples** for guidance

---

## 🗂️ Project Structure
```
exam-prep-chatbot/
├── assets/                  # Images and static files
├── data/                    # Uploaded PDF documents
│   ├── text_example_en.pdf
│   └── text_example_cn.pdf
├── gradio_helper.py         # UI logic
├── genai_helper.py          # LLM generation logic
├── ov_langchain_helper.py   # RAG + OpenVINO integration
├── rag_ui.py                # Gradio app launcher
├── llm_config.py            # Configurations for model & embeddings
├── llm-rag-langchain.ipynb  # Jupyter walkthrough
├── requirements.txt         # Dependencies
└── README.md                # Project guide (this file)
```

---

## 🧪 Example Interactions

### 🌐 English
**User**: _What are the main components of a RAG system?_

**Bot**:
```
A RAG system has three major components:

1. **Document Indexing**: Converts docs into embeddings.
2. **Retriever**: Finds relevant passages for a query.
3. **Generator**: Synthesizes a response using an LLM.

It bridges search and generation for precise, contextual answers.
```

### 🀄 Chinese
**User**: _什么是检索增强生成技术？_

**Bot**:
```
检索增强生成（RAG）结合了文档检索与语言模型生成。
它先将文档转为嵌入向量，通过检索找出相关段落，
再由大型语言模型生成自然语言回答，确保准确性与上下文一致性。
```

---

## 🧩 Configuration

Customize your model settings in `llm_config.py`:
```python
config = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "use_openvino": True,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200
}
```

---

## 📦 Dependencies

- [LangChain](https://github.com/hwchase17/langchain)
- [OpenVINO](https://www.openvino.ai/)
- [Gradio](https://www.gradio.app/)
- [Sentence Transformers](https://www.sbert.net/)
- FAISS, NumPy, PyPDF2, Jupyter



---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for more info.

---

## 🙏 Acknowledgements
- 🧠 [LangChain](https://github.com/hwchase17/langchain)
- 🚀 [OpenVINO](https://www.openvino.ai/)
- 💬 [Gradio](https://www.gradio.app/)
- 🤗 [Hugging Face](https://huggingface.co/)
- ❤️ Contributor:[Godreign Elgin Y](https://github.com/GodreignElgin)

