import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Function to dynamically select the model
def get_model_and_tokenizer(token_count):
    if token_count < 100:
        model_name = "bert-base-uncased"  # Small model for smaller inputs
    else:
        model_name = "bert-large-uncased"  # Large model for larger inputs

    st.write(f"**Selected Model:** {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to calculate token count
def calculate_token_count(prompt, tokenizer):
    tokenized_prompt = tokenizer(prompt, truncation=True, padding=True, max_length=512)
    return len(tokenized_prompt["input_ids"])

# Streamlit UI
st.title("Dynamic Model Selection for NLP Tasks")
st.write("Enter a text prompt below to evaluate and dynamically select the appropriate model (BERT-small or BERT-large) based on the token count.")

# Input prompt from user
prompt = st.text_area("Enter your text prompt:", "")

if st.button("Evaluate"):
    if not prompt.strip():
        st.error("Please enter a valid prompt!")
    else:
        with st.spinner("Processing..."):
            # Temporary tokenizer for token count calculation
            temp_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Calculate token count
            token_count = calculate_token_count(prompt, temp_tokenizer)
            st.write(f"**Token Count:** {token_count}")

            # Dynamically select the model
            model, tokenizer = get_model_and_tokenizer(token_count)

            # Initialize pipeline for prediction
            nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

            # Generate predictions
            predictions = nlp_pipeline(prompt)

            # Display predictions
            st.subheader("NER Predictions")
            if predictions:
                for pred in predictions:
                    st.write(f"**Token:** {pred['word']} | **Entity:** {pred['entity']} | **Score:** {pred['score']:.2f}")
            else:
                st.write("No entities detected in the prompt.")