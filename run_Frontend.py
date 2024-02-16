import os
import soundfile as sf
import torch
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torchaudio.transforms import Resample
from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download
from summarizer import Summarizer
import streamlit as st


# Define the repository name and filename
repo_name = "zicsx/Hindi-Punk"
filename = "Hindi-Punk-model.pth"

# Download the file
model_path = hf_hub_download(repo_id=repo_name, filename=filename)

# Load the state_dict and modify keys if necessary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cuda' if available

# ... (rest of the key handling logic)


# ASR script
def convert_audio_os(src, new_rate=16000):
    print("\nInside convert_audio_os")
    # Specify a temporary path for the converted audio
    temp_output_path = 'Hindi Audio/16Hz_Converted/temp_converted_audio_16kHz.wav'
    os.system(r'C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg -i "{}" -ac 1 -ar {} "{}"'.format(src, new_rate, temp_output_path))
    return temp_output_path

def parse_transcription(wav_file, chunk_size_seconds=5):
    # Load pretrained model
    processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
    model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")

    # Load audio
    audio_input, sample_rate = sf.read(wav_file)

    # Ensure the audio is mono
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)

    # Resample audio to 16 kHz if needed
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000) 
        audio_input = resampler(torch.tensor(audio_input, dtype=torch.float32)).numpy()


    # Calculate the number of chunks
    chunk_size_samples = int(chunk_size_seconds * 16000)
    num_chunks = int(np.ceil(len(audio_input) / chunk_size_samples))

    # Initialize an empty transcription
    full_transcription = ""

    # Process each chunk
    for i in range(num_chunks):
        start_idx = i * chunk_size_samples
        end_idx = min((i + 1) * chunk_size_samples, len(audio_input))
        chunk = audio_input[start_idx:end_idx]

        # Pad input values and return PyTorch tensor
        input_values = processor(chunk, sampling_rate=16000, return_tensors="pt").input_values

        # Inference
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Transcribe and append to the full transcription
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
        full_transcription += transcription + " "

    print("Full Transcription without Punctuation :", full_transcription)
    return full_transcription

# Punctuation script (from previous response)
class CustomTokenClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(CustomTokenClassifier, self).__init__()
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None

    def forward(self, hidden_states):
        if self.classifier:
            return self.classifier(hidden_states)
        else:
            return None

class PunctuationModel(nn.Module):
    def __init__(self, bert_model_name, punct_num_classes, hidden_size):
        super(PunctuationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.punct_classifier = CustomTokenClassifier(hidden_size, punct_num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        punct_logits = self.punct_classifier(hidden_states) if self.punct_classifier else None
        return punct_logits

# Initialize and load the model (from previous response)
model = PunctuationModel(
    bert_model_name='google/muril-base-cased',
    punct_num_classes=5,  # Number of punctuation classes (including 'O')
    hidden_size=768       # Hidden size of the BERT model
)

# Load the state_dict and modify keys if necessary (from previous response)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cuda' if available

# ... (rest of the key handling logic from previous response)

# Load the modified state_dict (from previous response)
model.load_state_dict(state_dict)

# Use the tokenizer associated with the model (from previous response)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="zicsx/Hindi-Punk", use_fast=True,
)

# Define Inference Functions (from previous response)
def predict_punctuation_capitalization(model, text, tokenizer):
    # Tokenize and truncate the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Determine the device to use (CPU or GPU)
    device = next(model.parameters()).device

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        punct_logits = model(**inputs)

    return punct_logits

# Function to map predictions to labels and combine them with the original text (from previous response)
def combine_predictions_with_text(text, tokenizer, punct_predictions, punct_index_to_label):
    # Convert logits to probabilities and get the indices of the highest probability labels
    punct_probs = torch.nn.functional.softmax(punct_predictions, dim=-1)
    punct_predictions = torch.argmax(punct_probs, dim=-1)

    # Tokenize the input text and get offset mappings
    encoded = tokenizer.encode_plus(text, return_tensors='pt', return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    offset_mapping = encoded['offset_mapping'][0].tolist()

    # Combine tokens with their predictions
    combined = []
    current_word = ''
    current_punct = ''
    for i, (token, punct) in enumerate(zip(tokens, punct_predictions.squeeze())):
        # Skip special tokens
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        # Remove "##" prefix from subword tokens
        if token.startswith("##"):
            token = token[2:]
        else:
            # If not the first token, add a space before starting a new word
            if current_word:
                combined.append(current_word + current_punct)
                current_word = ''
                current_punct = ''
        
        current_word += token

        # Update the current punctuation if predicted
        if punct_index_to_label[punct.item()] != 'O':
            current_punct = punct_index_to_label[punct.item()]

    # Append the last word and punctuation (if any) to the combined text
    combined.append(current_word + current_punct)

    return ' '.join(combined)

# Punctuation label to index mapping (from previous response)
punct_index_to_label = {0: '', 1: '!', 2: ',', 3: '?', 4: '।'}

# ASR + Punctuation + Summerization (Combining both scripts)

def main():
    st.title("Mahindra Finance MALT POC")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        # Save the uploaded file
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ASR: Perform ASR on the uploaded audio file
        st.header("Full Transcription without Punctuation")
        asr_transcription = parse_transcription("temp_audio.wav")
        st.write(asr_transcription)


        chunk_size = 512  # Adjust this based on the model's maximum sequence length
        text_chunks = [asr_transcription[i:i+chunk_size] for i in range(0, len(asr_transcription), chunk_size)]

        # Punctuation: Predict punctuation for the transcript
        st.header("Final Text with Punctuation")
        combined_chunks = []
        for chunk in text_chunks:
            punct_predictions = predict_punctuation_capitalization(model, chunk, tokenizer)
            combined_text = combine_predictions_with_text(chunk, tokenizer, punct_predictions, punct_index_to_label)
            combined_chunks.append(combined_text) 
            st.write(combined_text)

        # Combine the results from different chunks
        final_combined_text = ' '.join(combined_chunks)

        # Summarization: Generate summary for the final text
        st.header("Summary")
        summarizer = Summarizer()
        summary = summarizer(final_combined_text)
        st.write(summary)

        # Remove the temporary audio file
        os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()