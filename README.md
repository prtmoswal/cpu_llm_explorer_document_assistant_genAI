# cpu_llm_explorer_document_assistant_genAI
Developed an interactive Streamlit application enabling local execution and exploration of small Large Language Models (LLMs) like distilgpt2 and tinystories entirely on CPU, ensuring offline functionality

•	Implemented Retrieval Augmented Generation (RAG) capabilities, allowing users to upload and process custom .docx documents to provide dynamic context for LLM responses, enhancing factual accuracy and domain-specific generation.
•	Designed a user-friendly interface with features for model selection, prompt input, and adjustable generation parameters (e.g., max_new_tokens, temperature).
•	Explored the practical challenges and limitations of deploying larger quantized models (e.g., Phi-2, Gemma-2B) on local CPU environments.
•	Demonstrated proficiency in building end-to-end Generative AI applications with a focus on resource efficiency and accessibility for study and experimentation.
•	Technologies Used:
•	Python: Core programming language.
•	Streamlit: For interactive web application development.
•	Hugging Face Transformers/Diffusers: For integrating pre-trained LLMs.
•	FAISS: For efficient similarity search in RAG implementation (implied by faiss_index).
•	python-docx: For document processing.
•	os, tempfile: For file handling.

Youtube link for demo: https://youtu.be/7O6oAchsPR8
