import sys
import time
import pandas as pd
import re
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt, get_chat_template

# === Model Config ===
max_seq_length = 2048
dtype = None
load_in_4bit = True
model_path = ''
# === Load Model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    #fast_inference=True,

)
#print("Vllm loaded")


# === Load and Melt Prompts ===
df = pd.read_csv('')
#df=df.head(10)
print("Original columns:", df.columns.tolist())

# Only keep necessary columns and melt
persuasion_cols = [
    "Logical Appeal", "Authority Endorsement", "Misrepresentation",
    "Anchoring", "Priming", "Confirmation Bias"
]


long_df = df.melt(
    id_vars=["question"],
    value_vars=persuasion_cols,
    var_name="Technique",
    value_name="base_prompt"
)

# Add column for predictions
long_df["predictions_model"] = ""

# === Format Prompts for Chat Template ===
def format_conversation(row):
    return {
        "conversations": [
            {"role": "user", "content": row["base_prompt"]},
        ]
    }

def create_dataset(df):
    return Dataset.from_pandas(pd.DataFrame([format_conversation(row) for _, row in df.iterrows()]))

dataset = create_dataset(long_df)
dataset = standardize_sharegpt(dataset)
#tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, enable_thinking=True,) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# === Inference ===
def generate_response(model, dataset, tokenizer, data, output_file, batch_size=1, save_interval=1):
    results = []

    for i in range(0, len(dataset), batch_size):
        batch_samples = dataset[i:i+batch_size]
        print(f"Processing batch {i+1}/{len(dataset)}")

        batch_inputs = tokenizer.apply_chat_template(
            [sample for sample in batch_samples['conversations']],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            batch_outputs = model.generate(
                input_ids=batch_inputs,
                max_new_tokens=8000,
                use_cache=True,
                #temperature=0.5,
                #min_p=0.1
            )

        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Inference Time: {end_time - start_time:.2f}s")

        predicted_texts = tokenizer.batch_decode(batch_outputs)

        batch_results = []
        for predicted_text in predicted_texts:
            match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)<\|eot_id\|>", predicted_text, re.DOTALL)
            predicted_label = match.group(1).strip() if match else predicted_text.strip()
            if predicted_label == "":
                predicted_label = "Invalid Output"
            print(predicted_label)
            batch_results.append(predicted_label)

        results.extend(batch_results)

        if (i + batch_size) % save_interval == 0 or (i + batch_size) >= len(dataset):
            data.loc[:len(results)-1, "predictions_model"] = results
            data.to_csv(output_file, index=False)
            print(f"✅ Saved {len(results)} results → {output_file}")

    return results

# === Run Inference ===
output_file = ""
results = generate_response(model, dataset, tokenizer, long_df, output_file)

print(f"✅ All predictions saved to: {output_file}")
