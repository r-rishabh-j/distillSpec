from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import GKDTrainer, GKDConfig

config = {
    'student_model_id': 'HuggingFaceTB/SmolLM-360M-Instruct',
    'teacher_model_id': 'HuggingFaceTB/SmolLM-1.7B-Instruct',
    'output_dir': '../checkpoints/smol-cnn-rkl',
    'dataset_name': 'rishabhrj11/cnn_dailymail_512',
    'teacher_quantization': False,
    'train_config': {
        'beta': 1.0,
        'max_new_tokens': 128,
        'temperature': 0.7,
        'per_device_train_batch_size': 150,
        'gradient_accumulation_steps': 1,
        'gradient_checkpointing': False,
        'learning_rate': 7e-5,
        'lr_scheduler_type': 'cosine',
        'warmup_steps': 20,
        'fp16': False,
        'bf16': True,
        'logging_steps': 10,
        'num_train_epochs': 2,
        'save_strategy': 'steps',
        'save_steps': 20,
        'remove_unused_columns': False,
        'push_to_hub': True,
        'hub_model_id': 'rishabhrj11/distillspec-smollm-cnn-rkl',
        'save_total_limit': 3,
    }
}
config = {
    'student_model_id': 'Qwen/Qwen3-0.6B',
    'teacher_model_id': 'Qwen/Qwen3-4B-Instruct-2507',
    'output_dir': '../checkpoints/qwen-gsm-rkl-unquant',
    'dataset_name': 'openai/gsm8k',
    'teacher_quantization': False,
    'train_config': {
        'beta': 1.0,
        'max_new_tokens': 312,
        'temperature': 0.6,
        'per_device_train_batch_size': 24,
        'gradient_accumulation_steps': 1,
        'gradient_checkpointing': False,
        'learning_rate': 5e-5,
        'lr_scheduler_type': 'cosine',
        'warmup_steps': 10,
        'fp16': False,
        'bf16': True,
        'logging_steps': 10,
        'num_train_epochs': 1,
        'save_strategy': 'steps',
        'save_steps': 20,
        'remove_unused_columns': False,
        'push_to_hub': True,
        'hub_model_id': 'rishabhrj11/distillspec-qwen6-rkl-unquant',
        'save_total_limit': 3,
    }
}
# ==========================================
# 1. Configuration
# ==========================================
STUDENT_ID = config['student_model_id']
TEACHER_ID = config['teacher_model_id']
OUTPUT_DIR = config['output_dir']

# ==========================================
# 2. Load Tokenizer & Datasets
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

if config['dataset_name'] == 'openai/gsm8k':
    dataset = load_dataset("openai/gsm8k", "main", split='train[900:]')
else:
    dataset = load_dataset("rishabhrj11/cnn_dailymail_512", split='train').shuffle(seed=42)

def format_prompts(examples):
    formatted_prompts = []
    prompt_texts = examples["question"] if config['dataset_name'] == 'openai/gsm8k' else examples["article"]
    for prompt_text in prompt_texts:
        if config['dataset_name'] == 'openai/gsm8k':
            messages = [
                {"role": "system", "content": "In math word problem given by the user, reason step by step and put your final answer within \boxed{}."},
                {"role": "user", "content": prompt_text}
            ]
        else:
            messages = [
                        {"role": "system", "content": "Write a very short summary for the user's article."},
                        {"role": "user", "content": prompt_text}
                    ]
        formatted_prompts.append(messages)

    return {"messages": formatted_prompts}

dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

# ==========================================
# 3. Load Teacher (Int8 Quantization)
# ==========================================
bnb_config = None
if config['teacher_quantization']:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

print("Loading Teacher (Int8)...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    dtype=torch.float16
)
teacher_model.eval() # Ensure teacher is in eval mode

print("Loading Student (Base)...")
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    device_map="auto",
    use_cache=False,
    dtype=torch.bfloat16,
)

# ==========================================
# 5. GKD (DistillSpec) Configuration
# ==========================================
gkd_config = GKDConfig(
    lmbda=1.0,           # 1.0 = Purely On-Policy (Student generates data). 0.0 = Off-policy (Fixed dataset)
    output_dir=OUTPUT_DIR,
    **config['train_config'],
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,
)

trainer = GKDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=gkd_config,
    train_dataset=dataset,
    processing_class=tokenizer, # Replaces 'tokenizer' in newer TRL versions
)

print("Starting DistillSpec Training...")
trainer.train()

print(f"Training complete. Saved to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)