import os
import io
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import torch # Required for device mapping, even for CPU

# Imports for local LLMs (Hugging Face Transformers)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Configuration for Chunking and Models ---
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

# Embedding model name (for RAG-like context retrieval)
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Global variables for models (initialized once)
embedding_model_instance = None
# Dictionary to hold different loaded LLM instances by their type/name
local_llm_models = {}

# --- Model Initialization ---
def initialize_embedding_model():
    """Initializes the embedding model once."""
    global embedding_model_instance
    if embedding_model_instance is None:
        print("Initializing embedding model...")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model initialized.")

def initialize_local_llm(model_type: str):
    """
    Initializes a specific local generative LLM based on its type.
    Currently supports 'distilgpt2' and 'tinystories'.
    """
    global local_llm_models

    if model_type in local_llm_models:
        print(f"Local LLM '{model_type}' already loaded.")
        return

    print(f"Initializing local generative LLM: {model_type}...")
    try:
        if model_type == "distilgpt2":
            model_id = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Store model and tokenizer separately for more control over generation
            llm_instance = {
                "model": model,
                "tokenizer": tokenizer,
                "model_max_length": tokenizer.model_max_length,
                "device": "cpu"
            }

        elif model_type == "tinystories":
            model_id = "roneneldan/TinyStories-1M"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Store model and tokenizer separately for more control over generation
            llm_instance = {
                "model": model,
                "tokenizer": tokenizer,
                "model_max_length": tokenizer.model_max_length,
                "device": "cpu"
            }

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                "Supported types are 'distilgpt2', 'tinystories'."
            )

        # Move model to specified device (CPU in this case)
        llm_instance["model"].to(llm_instance["device"])
        local_llm_models[model_type] = llm_instance
        print(f"Local LLM '{model_type}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing local LLM '{model_type}': {e}")
        # Remove from dictionary if initialization failed
        if model_type in local_llm_models:
            del local_llm_models[model_type]
        raise # Re-raise to let Streamlit know there was an error

# --- Document Processing Functions (Same as before) ---

def extract_text_from_docx(docx_path: str) -> str:
    """Extracts text from a .docx file given its path."""
    doc = Document(docx_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    raw_text = "\n".join(full_text)

    # Basic cleaning
    cleaned_text = raw_text.replace("[Image 1]", "").replace("[Image 2]", "").strip()
    # Remove empty lines that might result from cleaning
    cleaned_text = "\n".join([line for line in cleaned_text.splitlines() if line.strip()])
    return cleaned_text

def chunk_text_by_words(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[str]:
    """Splits text into overlapping chunks based on words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words - chunk_overlap_words):
        chunk = " ".join(words[i : i + chunk_size_words])
        if chunk:
            chunks.append(chunk)
    return chunks

def load_and_process_document_for_context(docx_path: str):
    """
    Loads text from a .docx file, chunks it, and creates a FAISS index.
    This is for *context retrieval* for the generative model.
    Returns (faiss_index, list_of_chunks_text).
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Document not found at: {docx_path}")

    initialize_embedding_model()

    print(f"Processing document from: {os.path.basename(docx_path)} for context.")
    cleaned_text = extract_text_from_docx(docx_path)
    document_chunks_list = chunk_text_by_words(cleaned_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    if not document_chunks_list:
        print("No content extracted or chunks created from the document.")
        return None, []

    print(f"Creating embeddings for {len(document_chunks_list)} chunks...")
    chunk_embeddings = embedding_model_instance.encode(document_chunks_list, show_progress_bar=True)

    # Ensure embeddings are float32 for FAISS
    document_embeddings = np.array(chunk_embeddings).astype('float32')

    # Create FAISS index
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(document_embeddings)
    print("FAISS index created for context retrieval.")

    return faiss_index, document_chunks_list

# --- Generative AI Inference Function ---

def generate_text_with_context(model_type: str, prompt: str, faiss_index, document_chunks: list[str], top_k_chunks: int = 1, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
    """
    Generates text using the loaded local LLM, potentially incorporating retrieved context.
    Dispatches to the correct model based on model_type.
    """
    if model_type not in local_llm_models:
        return f"Local generative AI model '{model_type}' not loaded. Please load it first."

    # Extract model and tokenizer from the stored dictionary
    llm_info = local_llm_models[model_type]
    model = llm_info["model"]
    tokenizer = llm_info["tokenizer"]
    model_max_length = llm_info["model_max_length"]
    device = llm_info["device"]

    # 1. Retrieve relevant context if document is loaded
    context = ""
    if faiss_index is not None and document_chunks:
        initialize_embedding_model()
        print(f"Searching for relevant document chunks for prompt: {prompt}")
        prompt_embedding = embedding_model_instance.encode([prompt]).astype('float32')
        D, I = faiss_index.search(prompt_embedding, top_k_chunks)
        relevant_contexts = [document_chunks[i] for i in I[0]]
        context = "\n\n".join(relevant_contexts)
        print(f"Retrieved {len(relevant_contexts)} relevant contexts for generation.")

        # Craft the prompt to include context
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
    else:
        full_prompt = prompt # If no document, just use the direct prompt

    print(f"\n--- Sending to Local LLM ({model_type}) --- \nPrompt (truncated if necessary): {full_prompt[:500]}...\n--------------------------")

    try:
        # Tokenize the full prompt
        inputs = tokenizer.encode_plus(
            full_prompt,
            return_tensors='pt',
            truncation=True, # Ensure truncation if input is too long
            max_length=model_max_length, # Truncate to model's max length
            padding=False # No padding needed for single input for generate method
        ).to(device) # Move inputs to the correct device (CPU)

        input_length = inputs['input_ids'].shape[1]

        # Check if the input prompt already fills the model's context window
        if input_length >= model_max_length:
            return (f"**Warning:** Your prompt (including context) is {input_length} tokens, "
                    f"which is at or exceeds the model's maximum context length ({model_max_length}). "
                    "The model cannot generate new tokens beyond the input. Please try a shorter prompt or reduce document context.")

        # Generate text using the model's generate method directly
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens, # Explicitly use max_new_tokens for output length
            num_return_sequences=1,
            temperature=temperature,
            # Ensure pad_token_id is always a valid ID (fall back to eos_token_id if None)
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, # Explicitly use EOS token
            do_sample=True if temperature > 0 else False,
            top_k=50, # Limit sampling to top 50 probable tokens
            top_p=0.9, # Limit sampling to tokens with cumulative probability of 0.9
            repetition_penalty=1.2 # ADDED: Penalize repetition
        )

        # Decode the generated sequences
        generated_text_with_prompt = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Remove the original prompt to get only the newly generated text
        # Be careful with how much of the prompt is exactly reproduced by the model
        if generated_text_with_prompt.startswith(full_prompt):
            generated_text = generated_text_with_prompt[len(full_prompt):].strip()
        else:
            # Fallback if the generated text doesn't perfectly start with the prompt
            generated_text = generated_text_with_prompt.strip()

        # Simple post-processing to reduce obvious repetitions if any are observed (for smaller models)
        if generated_text and len(generated_text) > 50 and generated_text.count(generated_text[:50]) > 2:
            generated_text = generated_text.split(generated_text[:50])[0] + "..." # Truncate after first repetition

        print(f"\n--- Local LLM Response ({model_type}) --- \n{generated_text[:500]}...\n--------------------------")
        return generated_text

    except Exception as e:
        print(f"Error during local LLM generation for {model_type}: {e}")
        # Provide a more informative error message to the user
        return (f"Error generating content from local LLM ({model_type}): {e}. "
                "This might be due to incompatible model setup, insufficient resources, "
                "or an issue with the prompt. Please check the terminal for full error details.")
