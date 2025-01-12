from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

dataset = load_dataset("your-dataset")
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
model = BertForQuestionAnswering.from_pretrained("bert-base-german-cased")

def preprocess(examples):
    inputs = tokenizer(examples["question"], examples["context"], truncation=True, padding=True, max_length=512)
    return inputs

tokenized_data = dataset.map(preprocess, batched=True)
training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", learning_rate=2e-5, num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_data["train"], eval_dataset=tokenized_data["validation"])
trainer.train()
MODEL_PATH=models/model.pth
