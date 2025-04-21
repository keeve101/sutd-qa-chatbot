import argparse
import json

from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

from unsloth import FastLanguageModel

from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """
### Instruction:
You are a Singapore University of Technology (SUTD) website chatbot to answer questions from prospective students about SUTD. Answer the question from the user using relevant context.

### Input:
{question}

### Response:
"""

PROMPT_WITH_RAG_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately answers the question.

### Instruction:
You are a Singapore University of Technology (SUTD) website chatbot to answer questions from prospective students about SUTD. Answer the question from the user using relevant context.

### Input:
{question}

### Context:
{context}

### Response:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["question"],
)

PROMPT_WITH_RAG = PromptTemplate(
    template=PROMPT_WITH_RAG_TEMPLATE,
    input_variables=["context", "question"],
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="keeve101/sutd-qa-dataset")
    parser.add_argument("--use_rag", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    dataset_path = args.dataset_path

    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
    )

    generation_kwargs = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
        "max_new_tokens": 256,
    }

    if args.use_rag:
        vector_store_dir = "vector_store_v2"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vector_store = FAISS.load_local(
            vector_store_dir, embeddings, allow_dangerous_deserialization=True
        )

        document_retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 5}
        )

        prompt_template = PROMPT_WITH_RAG
    else:
        prompt_template = PROMPT

    split = "test"

    test = concatenate_datasets(
        [
            load_dataset(dataset_path, config_name, split=split)
            for config_name in get_dataset_config_names(dataset_path)
        ]
    )

    output_file = model_name.split("/")[-1] + "_" + split + ".jsonl"

    for example in test:
        question = example["question"]
        answer = example["answer"]

        if args.use_rag:
            context = document_retriever.invoke(question)

            prompt = prompt_template.format(question=question, context=context)
        else:
            prompt = prompt_template.format(question=question)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        results = model.generate(**inputs, **generation_kwargs)
        
        input_length = inputs["input_ids"].shape[1]
        generated_ids = results[0][input_length:]
        predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        with open(output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "question": question,
                        "reference_answer": answer,
                        "prompt": prompt,
                        "predicted_answer": predicted_answer,
                        "model_name": model_name,
                        "with_rag": args.use_rag,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
