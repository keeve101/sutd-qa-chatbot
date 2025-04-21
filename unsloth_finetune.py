from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

from unsloth import FastLanguageModel
from unsloth.trainer import SFTTrainer, TrainingArguments, is_bfloat16_supported

from langchain.prompts import PromptTemplate

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=max_seq_length,
)

peft_model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=0,
    use_rslora=False,
    loftq_config=None,
)

dataset_path = "keeve101/sutd-qa-dataset"

splits = ["train", "validation", "test"]

config_names = get_dataset_config_names(dataset_path)

datasets = {
    config_name: load_dataset(dataset_path, config_name) for config_name in config_names
}

dataset = [
    concatenate_datasets(
        [datasets[config_name][split] for config_name in datasets.keys()]
    )
    for split in splits
]

train, validation, test = dataset

PROMPT_TEMPLATE = """
### Instruction:
You are a Singapore University of Technology (SUTD) website chatbot to answer questions from prospective students about SUTD. Answer the question from the user using relevant context.

### Input:
{question}

### Response:
{answer}
"""

PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template=PROMPT_TEMPLATE,
)


def batched_format_prompt_func(examples):
    texts = []

    questions = examples["question"]
    answers = examples["answer"]

    for question, answer in zip(questions, answers):
        text = PROMPT.format(question=question, answer=answer) + tokenizer.eos_token

        texts.append(text)

    return {"texts": texts}


train = train.map(batched_format_prompt_func, batched=True)
validation = validation.map(batched_format_prompt_func, batched=True)

output_dir = "models"

trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    train_dataset=train,
    eval_dataset=validation,
    dataset_text_field="texts",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        output_dir=output_dir,
        report_to="none",
    ),
)

trainer.train()

peft_model.push_to_hub("keeve101/llama-3.2-1B-sutdqa")
tokenizer.push_to_hub("keeve101/llama-3.2-1B-sutdqa")
