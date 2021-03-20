from transformers import pipeline
from transformers import set_seed

set_seed(0)

generator = pipeline('text-generation',
                     model='../models/gpt2-lambada',
                     tokenizer='gpt2',
                     config={'max_length':1024},
                     device=0)

PROMPT = "You are Larry, a knight living in the kingdom of Larion. You have a steel longsword and a wooden shield. You are on a quest to defeat the evil dragon of Larion. You've heard he lives up at the north of the kingdom. You set on the path to defeat him and walk into a dark forest. As you enter the forest you see"

greedy_result = generator(PROMPT, max_length=1024, early_stopping=True)
print("Greedy search resulted in\n> {}".format(greedy_result[0]['generated_text']))
print()

beam_result = generator(PROMPT, max_length=1024, early_stopping=True,
                        num_return_sequences=5,
                        num_beams=5)
print("Beam search resulted in")
for text in beam_result:
  print("> {}".format(text['generated_text']))
print()
sample_result = generator(PROMPT, max_length=1024, early_stopping=True,
                          num_return_sequences=5,
                          do_sample=True, top_k=0)
print("Sample search resulted in")
for text in sample_result:
  print("> {}".format(text['generated_text']))
print()

sample_temp_result = generator(PROMPT, max_length=1024, early_stopping=True,
                               num_return_sequences=5,
                               do_sample=True, top_k=0, temperature = 0.7)
print("Sample search with temperature resulted in")
for text in sample_temp_result:
  print("> {}".format(text['generated_text']))
print()

sample_k_result = generator(PROMPT, max_length=1024, early_stopping=True,
                            num_return_sequences=5,
                            do_sample=True, top_k=50)
print("Sample search with top_k resulted in")
for text in sample_k_result:
  print("> {}".format(text['generated_text']))
print()

sample_p_result = generator(PROMPT, max_length=1024, early_stopping=True,
                            num_return_sequences=5,
                            do_sample=True, top_p=0.92, top_k=0)
print("Sample search with top_p resulted in")
for text in sample_p_result:
  print("> {}".format(text['generated_text']))
print()

sample_pk_result = generator(PROMPT, max_length=1024, early_stopping=True,
                             num_return_sequences=5,
                             do_sample=True, top_p=0.95, top_k=50)
print("Sample search with top_p and top_k resulted in")
for text in sample_pk_result:
  print("> {}".format(text['generated_text']))
print()
