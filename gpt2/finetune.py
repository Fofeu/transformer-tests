import transformers
import datasets

from datasets import load_from_disk

train_dataset = load_from_disk("../datasets/lambada-gpt2")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().train()

training_args = TrainingArguments(
  output_dir="./../models/gpt2-lambada", #The output directory
  overwrite_output_dir=True, #overwrite the content of the output directory
  num_train_epochs=3, # number of training epochs
  per_device_train_batch_size=1, # batch size for training
  gradient_accumulation_steps=128,  # accumulate gradients to increase batchsize
  per_device_eval_batch_size=16,  # batch size for evaluation
  eval_steps = 400, # Number of update steps between two evaluations.
  save_steps=800, # after # steps model is saved
  warmup_steps=500,# number of warmup steps for learning rate scheduler
  fp16=True, # use mixed precision
  fp16_opt_level='O1', # O0=no mp, O1=fp16 where it matters, O2=nearly everything is fp16, O3=everything is fp16
)

trainer = Trainer(
  model=model,
  tokenizer=tokenizer,
  args=training_args,
  train_dataset=train_dataset,
)

import torch

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1.0)

trainer.train()

trainer.save_model()
