# LoRA Fine-tuning SOP

## Outline

- Introduction
- Steps in Fine-tuning
- Prep for Training Data

## Introduction

Since it’s expensive to retrain every weight in LLM, other techniques were developed to achieve similar results. Here Low-rank Adaptation of LLM (LoRA) and Quantization reduce the overall training time. In simple terms, LoRA generates an extra weight matrix based on the original weight matrix from the LLM without actually changing the original matrix. When inference, the two matrices were merged into a fine-tuned matrix. Quantization, on the other hand, is more straightforward. It shortens the number of bits in which the LLM’s weights store. An 8-bit quantization cuts the memory in halves and a 4-bit quantization further reduces the memory used by 4 times.

## Steps in fine-tuning

1. Choose a pre-trained LLM.
2. Configure Quantization.
3. Load the LLM in memory.
4. Configure LoRA (hyper)parameters
5. Get PEFT model with LLM and LoRA
6. Set up training arguments for Trainer
7. Save the model after training/fine-tuning.

Steps 1-3
Here I load the model in 4 bit.

```py
MODEL_NAME = "FlagAlpha/Llama2-Chinese-7b-Chat"
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    # trust_remote_code=True,
    # local_files_only=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
```

Step 4-5 Freeze most of the LLM’s parameters

```py
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
lora_target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]


config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
```

Step 6 Set up Trainer for fine-tuning

```py
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    max_steps=80,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

%%time
trainer.train()
```

## Prep for Training Data

1. Save data into csv format.
2. Load csv into pandas DataFrame.
3. Convert pandas DataFrame into DataSet for huggingface.
4. Concatenate prompt and completion into one string for training.
5. Tokenize the string.

Step 1. Save as csv format.

```py
data = [
      {
            "question": "如何跳轉到指定工站？",
            # "role": "Service Centre",
            "answer": "需要登錄管理員帳號, 服務中心後臺->進度管控->進度維護, 進入【保修項目】分頁, 勾選對應產品序號, 下拉【維修操作】點擊【進度維護】, 選擇【跳轉至指定工站】即可進行跳轉."
      },
      {
            "question": "所收返修品和保修申請內容不符如何處理？",
            # "role": "Service Centre",
            "answer": "服務中心後台->貨物收發->收發管理, 在【待收貨】分頁進行收貨時, 如若出現貨不符實的情況, 可選取該產品點擊【刪除】, 該產品資訊將會返回給客戶確認, 客戶可重新提交申請."
      },
]
```

```py
import pandas as pd
df = pd.DataFrame(data) 
df.to_csv("./data/warreconn_data_cn.csv", index=False)
```

Step 2 - 3.

```py
data = pd.read_csv("./data/warreconn_data_cn.csv")
from datasets import Dataset
dataset = Dataset.from_pandas(data)
```

Step 4 - 5.
```py
def generate_prompt(data_point):
    return f"""
: {data_point["question"]}
: {data_point["answer"]}
""".strip()


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt

train_data = dataset.shuffle().map(generate_and_tokenize_prompt)
```

- Resulting Dataset
```py
Dataset({
    features: ['question', 'answer', 'input_ids', 'attention_mask'],
    num_rows: 6
})
```

```py
dataset[0]

{'question': '如何跳轉到指定工站？',
 'answer': '需要登錄管理員帳號, 服務中心後臺->進度管控->進度維護, 進入【保修項目】分頁, 勾選對應產品序號, 下拉【維修操作】點擊【進度維護】, 選擇【跳轉至指定工站】即可進行跳轉.'}
```

## Reference

https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/quantization#fp4-quantization

https://huggingface.co/docs/peft/conceptual_guides/lora