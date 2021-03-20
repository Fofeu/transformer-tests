import transformers
import datasets
from datasets import load_dataset

# We start from a dataset available from huggingface
dataset = load_dataset("lambada")

# Let's only care about the train dataset for now
train_dataset = dataset["train"]

from transformers import AutoTokenizer
# We use the default gpt2 tokenizer and define the padding token as the EOS token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# In a first pass, we simply tokenize the whole string
def tokenize(x):
  result = tokenizer(x["text"])
  return result

# Some required values
pad_token = tokenizer.pad_token_id
block_size = tokenizer.model_max_length

# Now we split the tokenized into sequences of length ``block_size''
def group_texts(examples):
  # We don't want to "pollute" different text between each other
  # So instead of concatenating everything together and then produce block_sized examples,
  # only split individual examples into chuncks of length block_size

  result = {
    k: sum([ [(x[i:i+block_size] if len(x[i:i+block_size]) == block_size else
               x[i:i+block_size] + [pad_token]*(block_size-len(x[i:i+block_size])))
              for i in range(0, len(x), block_size)] for x in t ], [])
    for k, t in examples.items()
  }

  # Prediction target is the same as output target
  result["labels"] = result["input_ids"].copy()
  return result

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text', 'domain'])

train_dataset = train_dataset.map(group_texts, batched=True)

train_dataset.save_to_disk("./lambada-gpt2")
