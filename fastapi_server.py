from unsloth import FastLanguageModel

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AsyncTextIteratorStreamer,
)
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


import asyncio
import torch
import json

vector_store_dir = "vector_store_v2"

base_model_name = "unsloth/Llama-3.2-1B"
finetuned_model_name = "keeve101/llama-3.2-1B-sutdqa"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = FAISS.load_local(
    vector_store_dir, embeddings, allow_dangerous_deserialization=True
)

document_retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 5}
)

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["question"],
)

PROMPT_WITH_RAG = PromptTemplate(
    template=PROMPT_WITH_RAG_TEMPLATE,
    input_variables=["context", "question"],
)

max_seq_length = 2048

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
)

finetuned_model, tokeinzer = FastLanguageModel.from_pretrained(
    finetuned_model_name,
    max_seq_length=max_seq_length,
)

app = FastAPI()


class GenerationRequest(BaseModel):
    query: str
    model: str
    use_rag: bool


@app.post("/generate")
async def generate(request: GenerationRequest):
    query = request.query
    model_name = request.model
    use_rag = request.use_rag

    prompt_kwargs = {"question": query}

    if use_rag:
        prompt = PROMPT_WITH_RAG
        retrieved_documents = document_retriever.invoke(query)
        context = "\n\n---------------\n\n".join(
            [
                f"{doc.page_content}\n{json.dumps(doc.metadata)}"
                for doc in retrieved_documents
            ]
        )

        prompt_kwargs.update({"context": context})
    else:
        prompt = PROMPT

    if model_name == "llm_base":
        model = base_model
    else:
        model = finetuned_model

    inputs = tokenizer(prompt.format(**prompt_kwargs), return_tensors="pt").to("cuda")

    streamer = AsyncTextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
        "max_new_tokens": 256,
    }

    async def generate_tokens():
        await asyncio.to_thread(model.generate, **generation_kwargs)

    asyncio.create_task(generate_tokens())

    async def token_stream():
        async for token in streamer:
            yield token
            await asyncio.sleep(0.001)

    return StreamingResponse(token_stream(), media_type="text/plain")
