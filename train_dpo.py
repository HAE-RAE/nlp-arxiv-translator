import random
import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

set_seed(42)
model_name = "Translation-EnKo/gukbap-gemma-2-9b-it-translation-general1.2m-en-ko-e1-b16-trc313eval45-e2-b16"
dataset_name = "Translation-EnKo/nlp-arxiv-translation-dpo-with-math-10k"

epoch = 2
batch_size = 1
gradient_accm = 4
lr = 7e-7
seq_len = 4096

out_mname = model_name.split('/')[-1]
out_dname = dataset_name.split('/')[-1]
total_batch_size = batch_size * gradient_accm
output_path = f"{out_mname}-{out_dname}-e{epoch}-b{total_batch_size}-lr{lr}-{seq_len}"

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir=f"./trained_model/{output_path}",
    evaluation_strategy="steps",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    bf16=True,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accm,
    per_device_eval_batch_size=batch_size,
    logging_steps=1,
    learning_rate=lr,
    do_eval=True,
    eval_steps=50,
    num_train_epochs=epoch,
    save_steps=100,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    deepspeed="config/ds_config_stage2.json",
    beta=0.1,
    max_prompt_length=seq_len // 2,
    max_length=seq_len,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
dataset = load_dataset(f'{dataset_name}-chat-gemma')
print(dataset)
trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    peft_config=peft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model()