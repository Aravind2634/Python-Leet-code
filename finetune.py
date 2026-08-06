from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# -------------------------------------------------------
# Step 1: Load a pre-trained model
# -------------------------------------------------------
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# -------------------------------------------------------
# Step 2: Create a small training dataset
# -------------------------------------------------------
data = [
    {
        "text": "Question: What is AI?\nAnswer: AI stands for Artificial Intelligence."
    },
    {
        "text": "Question: What is Python?\nAnswer: Python is a programming language."
    },
    {
        "text": "Question: What is Machine Learning?\nAnswer: Machine Learning enables computers to learn from data."
    },
]

dataset = Dataset.from_list(data)

# -------------------------------------------------------
# Step 3: Tokenize the dataset
# -------------------------------------------------------
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

    # Labels are the same as input_ids for causal language modeling
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

# -------------------------------------------------------
# Step 4: Training configuration
# -------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

# -------------------------------------------------------
# Step 5: Fine-tune
# -------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# -------------------------------------------------------
# Step 6: Save the model
# -------------------------------------------------------
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("✅ Fine-tuning completed successfully!")
