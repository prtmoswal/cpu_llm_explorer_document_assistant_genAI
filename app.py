import streamlit as st
import os
import tempfile
# Import functions from your backend.py file
from backend import (
    initialize_local_llm,
    load_and_process_document_for_context,
    generate_text_with_context,
    local_llm_models # To check which models are loaded
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="GenAI Study: CPU LLM Explorer", # Renamed page title
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 GenAI Study: CPU LLM Explorer & Document Assistant") # Renamed app title
st.markdown("Run powerful language models directly on your machine, with optional document context!")

# --- Session State Management ---
# Initialize session state variables if they don't exist
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'current_user_prompt' not in st.session_state:
    st.session_state.current_user_prompt = "Tell me a short story about a talking animal."


# Create tabs for better organization
main_tab, usage_tab = st.tabs(["💬 Chat & RAG", "💡 Usage Tips & Examples"])

with main_tab:
    # --- Sidebar for Model Selection and Initialization ---
    st.sidebar.header("1. Model Configuration")
    model_options = ["distilgpt2", "tinystories"]
    chosen_model_type = st.sidebar.selectbox(
        "Select a Local LLM:",
        model_options,
        index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    )

    # Initialize button for the selected model
    if st.sidebar.button(f"🚀 Initialize {chosen_model_type}"):
        try:
            with st.spinner(f"Loading {chosen_model_type} (this may take a moment)..."):
                initialize_local_llm(chosen_model_type)
            st.session_state.selected_model = chosen_model_type
            st.sidebar.success(f"{chosen_model_type} loaded successfully!")
        except FileNotFoundError as fnf_err:
            st.sidebar.error(f"Error: {fnf_err}. Please ensure the model file is in the correct path.")
        except Exception as e:
            st.sidebar.error(f"Error initializing {chosen_model_type}: {e}")

    # Display current model status
    if st.session_state.selected_model and st.session_state.selected_model in local_llm_models:
        st.sidebar.write(f"**Current Model:** `{st.session_state.selected_model}` (Ready)")
    else:
        st.sidebar.write("**Current Model:** No model loaded.")

    # --- Sidebar for Document Upload (RAG Context) ---
    st.sidebar.header("2. Document for Context (Optional)")
    uploaded_file = st.sidebar.file_uploader("Upload a .docx document (for RAG):", type=["docx"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location to be read by docx
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if st.sidebar.button("📄 Process Document for RAG"):
            if st.session_state.selected_model not in local_llm_models:
                st.sidebar.warning("Please initialize a model first before processing documents.")
            else:
                try:
                    with st.spinner("Processing document and creating embeddings..."):
                        faiss_index, document_chunks = load_and_process_document_for_context(tmp_file_path)

                    st.session_state.faiss_index = faiss_index
                    st.session_state.document_chunks = document_chunks
                    st.session_state.document_name = uploaded_file.name
                    st.sidebar.success(f"Document '{uploaded_file.name}' processed with {len(document_chunks)} chunks!")
                except Exception as e:
                    st.sidebar.error(f"Error processing document: {e}")
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)

    # Display current document status
    if st.session_state.document_name:
        st.sidebar.write(f"**Current Document:** `{st.session_state.document_name}` (Ready for RAG)")
    else:
        st.sidebar.write("**Current Document:** None loaded for RAG.")

    # --- Main Area for Prompt Input and Generation ---
    st.header("3. Ask Your Question")

    # Recommended Prompts Section
    st.subheader("Try a Recommended Prompt:")
    col_prompts_1, col_prompts_2 = st.columns(2)

    distilgpt2_prompts = {
        "Short poem about a [flower]": "Write a short, happy poem about a [flower].",
        "Complete: 'The old house stood on a hill, overlooking the [landscape].'": "Complete the sentence: 'The old house stood on a hill, overlooking the [landscape].'",
        "Describe a [weather] morning in two sentences.": "Describe a [weather] morning in two sentences.",
        "Fact about [historical figure]": "Tell me a fact about [historical figure].",
        "Start a story: a [object] came to life.": "Start a story about a [object] that came to life.",
        "Biggest [planet] in solar system?": "What is the biggest [planet] in our solar system?",
        "Slogan for a [product type] company.": "Give me a short slogan for a [product type] company.",
        "Email subject line: [event].": "Write a quick email subject line about a [event].",
        "Explain [simple concept] in one sentence.": "Explain [simple concept] in one sentence.",
        "Continue: 'If I had a superpower, it would be [power] because...'": "Continue the thought: 'If I had a superpower, it would be [power] because...'"
    }

    tinystories_prompts = {
        "A little [animal] went to the park.": "A little [animal] went to the park.",
        "The [color] car drove fast.": "The [color] car drove fast.",
        "A [character] found a [object].": "A [character] found a [object].",
        "The [noun] was very [adjective].": "The [noun] was very [adjective].",
        "A small [bird] flew high.": "A small [bird] flew high.",
        "The [toy] played with the [child].": "The [toy] played with the [child].",
        "A [animal] ate a [food].": "A [animal] ate a [food].",
        "The [person] smiled at the [object].": "The [person] smiled at the [object].",
        "A big [animal] lived in a [place].": "A big [animal] lived in a [place].",
        "The [animal] had a [feature].": "The [animal] had a [feature]."
    }

    with col_prompts_1:
        selected_distilgpt2_prompt_key = st.selectbox(
            "For `distilgpt2`:",
            ["Select a prompt...", *distilgpt2_prompts.keys()],
            key="distilgpt2_prompt_select"
        )
        if selected_distilgpt2_prompt_key != "Select a prompt...":
            st.session_state.current_user_prompt = distilgpt2_prompts[selected_distilgpt2_prompt_key]

    with col_prompts_2:
        selected_tinystories_prompt_key = st.selectbox(
            "For `tinystories`:",
            ["Select a prompt...", *tinystories_prompts.keys()],
            key="tinystories_prompt_select"
        )
        if selected_tinystories_prompt_key != "Select a prompt...":
            st.session_state.current_user_prompt = tinystories_prompts[selected_tinystories_prompt_key]

    user_prompt = st.text_area(
        "Or enter your custom prompt here (edit selected prompt directly):",
        value=st.session_state.current_user_prompt,
        height=150,
        key="main_user_prompt_text_area" # Added a key to avoid duplicate widget error
    )

    st.subheader("Generation Settings")
    col1, col2 = st.columns(2)
    with col1:
        gen_max_new_tokens = st.slider(
            "Max New Tokens (output length):",
            min_value=20, max_value=500, value=100, step=10,
            help="Controls the maximum number of new tokens (words/sub-words) the model will generate."
        )
    with col2:
        gen_temperature = st.slider(
            "Temperature (creativity):",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="Controls the randomness of the output. Lower values (e.g., 0.1) are more deterministic, higher values (e.g., 0.9) are more creative/random."
        )

    if st.button("✨ Generate Response"):
        if not st.session_state.selected_model:
            st.error("Please select and initialize a model in the sidebar first!")
        else:
            # Prepare the full prompt for generation, incorporating RAG context if available
            final_prompt = user_prompt
            if st.session_state.faiss_index and st.session_state.document_chunks:
                st.info(f"Using document '{st.session_state.document_name}' as context.")
                # The backend's generate_text_with_context function will handle
                # integrating the context into the prompt if faiss_index and document_chunks are provided.

            with st.spinner(f"Generating response using {st.session_state.selected_model}..."):
                response = generate_text_with_context(
                    model_type=st.session_state.selected_model,
                    prompt=final_prompt, # Pass the user's base prompt
                    faiss_index=st.session_state.faiss_index, # Pass RAG index
                    document_chunks=st.session_state.document_chunks, # Pass document chunks
                    max_new_tokens=gen_max_new_tokens,
                    temperature=gen_temperature
                )
                st.subheader("Generated Response:")
                st.write(response)

with usage_tab:
    st.header("💡 Usage Tips & Examples for Small Models")
    st.markdown("""
    These models (`distilgpt2`, `tinystories`) are **very small** compared to larger LLMs (e.g., GPT-3.5, Llama 2).
    They are designed for efficiency and can run on limited resources, but they have significant limitations in terms of:
    * **Knowledge:** Their understanding of facts is limited to their training data.
    * **Reasoning:** They struggle with complex logic, summarization, or answering nuanced questions.
    * **Coherence:** They can sometimes produce repetitive or nonsensical output, especially for longer generations.

    **Key to success:** Be specific, concise, and manage your expectations!
    """)

    # --- NEW: Model Comparison Chart ---
    st.subheader("📊 Model Comparison Chart")
    st.markdown("""
    Here's a comparison of the different model families we've discussed, highlighting their general characteristics.
    """)

    # Data for the comparison table
    model_comparison_data = {
        "Model Family": [
            "TinyStories (e.g., 1M)",
            "DistilGPT2 (82M)",
            "Gemma 2B (2B)",
            "Phi-2 (2.7B)"
        ],
        "Parameters": [
            "~1 Million",
            "82 Million",
            "~2 Billion",
            "~2.7 Billion"
        ],
        "Strengths / Use Cases": [
            "Extremely simple, child-like story generation with limited vocabulary. Ideal for exploring minimal language capabilities.",
            "Short text completion, simple creative prompts, basic Q&A (general text). Good for resource-constrained systems.",
            "Text generation, summarization, extraction, question answering, reasoning. Designed for resource-limited devices (laptops, mobile) and good performance for its size.",
            "QA, chat, code generation (primarily Python), common sense, language understanding. Strong performance for a relatively small LLM."
        ],
        "Limitations": [
            "Very limited vocabulary, struggles with complex instructions, abstract concepts, or coherence beyond very simple sentences. Not for general chat or factual tasks.",
            "Limited contextual understanding, lack of fact-checking, prone to repetition, struggles with complex instructions, summarization, or detailed reasoning.",
            "Like other LLMs, may inherit biases; potential for hallucinations; general LLM limitations. Requires more resources than DistilGPT2/TinyStories.",
            "May generate inaccurate code/facts, unreliable for complex/nuanced instructions, primarily standard English, potential biases/toxicity, can be verbose. Requires more resources than smaller models."
        ],
        "Reference": [
            "[Hugging Face - TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)",
            "[Hugging Face - distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2)",
            "[Hugging Face - google/gemma-2b](https://huggingface.co/google/gemma-2b)",
            "[Hugging Face - microsoft/phi-2](https://huggingface.co/microsoft/phi-2)"
        ]
    }

    # Format the data for a Markdown table
    table_headers = "| " + " | ".join(model_comparison_data.keys()) + " |"
    table_separator = "| " + " | ".join(["---"] * len(model_comparison_data.keys())) + " |"

    table_rows = []
    num_models = len(model_comparison_data["Model Family"])
    for i in range(num_models):
        row = [str(model_comparison_data[key][i]) for key in model_comparison_data.keys()]
        table_rows.append("| " + " | ".join(row) + " |")

    full_table_markdown = "\n".join([table_headers, table_separator] + table_rows)

    st.markdown(full_table_markdown)

    st.subheader("`distilgpt2` (General-Purpose Text Generation)")
    st.markdown("""
    `distilgpt2` is a distilled version of GPT-2, making it faster and smaller. It's best for:
    * **Short completions:** Finishing sentences or short paragraphs.
    * **Creative writing prompts:** Getting a small burst of text.
    * **Simple questions:** If the answer is straightforward and common.
    * **Avoid:** Complex summarization, deep factual retrieval, or multi-turn conversations.
    """)
    st.markdown("**Example Prompts & Expected Output:**")

    st.markdown("---")
    st.markdown("### Example 1 (Short Completion)")
    st.markdown("**Prompt:** `The quick brown fox jumped over the lazy`")
    st.markdown("**Expected Output (distilgpt2):** `...dog, and then ran off into the woods. The fox was very fast and agile.`")
    st.markdown("*(Note: Output might vary, but should be a logical short continuation)*")

    st.markdown("### Example 2 (Simple Creative Prompt)")
    st.markdown("**Prompt:** `Write a haiku about autumn leaves.`")
    st.markdown("**Expected Output (distilgpt2):** `...Red and gold they fall,\nDancing softly on the breeze,\nWinter's breath is near.`")
    st.markdown("*(The model tries to follow the instruction, but might not perfectly adhere to haiku rules)*")

    st.markdown("### Example 3 (Simple Question - No RAG needed)")
    st.markdown("**Prompt:** `What is the capital of France?`")
    st.markdown("**Expected Output (distilgpt2):** `...The capital of France is Paris. Paris is a city in France.`")
    st.markdown("*(Will likely repeat itself slightly)*")

    st.markdown("---")

    st.subheader("`tinystories` (Children's Stories)")
    st.markdown("""
    `tinystories` was trained on synthetic data of very simple, child-like stories with an extremely limited vocabulary.
    It's exclusively good for:
    * **Generating very short, simple children's stories.**
    * **Exploring basic narrative patterns.**
    * **Avoid:** Anything complex, factual, or outside the realm of simple, almost nonsensical, narratives. Expect repetitive and often simplistic language.
    """)
    st.markdown("**Example Prompts & Expected Output:**")

    st.markdown("---")
    st.markdown("### Example 1 (Simple Story Prompt)")
    st.markdown("**Prompt:** `A brave little bear went to the forest.`")
    st.markdown("**Expected Output (tinystories):** `...He saw a big tree. He climbed up the tree. He saw a little bird. The bird sang a song.`")
    st.markdown("*(Very simple sentences, repetitive structure)*")

    st.markdown("### Example 2 (Animal Character Prompt)")
    st.markdown("**Prompt:** `Once there was a cat named Mimi.`")
    st.markdown("**Expected Output (tinystories):** `...Mimi liked to play with a ball. She rolled the ball. The ball went fast.`")
    st.markdown("*(Focuses on simple actions and objects)*")

    st.markdown("---")

    st.subheader("Tips for Using RAG (Document Context)")
    st.markdown("""
    For `distilgpt2` and `tinystories`, the RAG feature (using uploaded documents) has **limited effectiveness** due to their small context window.
    * **Keep questions very short:** "Who is Purab?" (if Purab is in the document).
    * **Expect basic answers:** They might extract a phrase but won't summarize complex paragraphs well.
    * **Reduce `Max New Tokens`:** Especially when using RAG, set `Max New Tokens` to a very low value (e.g., 20-50) to allow the model to try and complete the answer before its context window is full.
    """)
    st.warning("If you need more advanced summarization, reasoning, or accurate long-form generation from documents, consider using a larger model like a quantized Phi-2 (if you enable the `phi-2-gguf` option again and have `llama-cpp-python` installed).")
