import json
import os
import pandas as pd

model_names = [
    "keeve101/llama-3.2-1B-sutdqa",
    "unsloth/Llama-3.2-1B",
]

qa_dict = {}

for model_name in model_names:
    base_model_name = model_name.split("/")[-1]

    for f in os.listdir():
        if f.endswith(".jsonl"):
            if base_model_name in f:
                with open(f, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        question = entry["question"]
                        ground_truth = entry["reference_answer"]
                        predicted = entry["predicted_answer"]
                        
                        if question not in qa_dict:
                            qa_dict[question] = {
                                "question": question,
                                "ground_truth": ground_truth
                            }
                        
                        column_name = model_name + "_rag" if entry["with_rag"] else model_name + "_no_rag"
                        
                        qa_dict[question][column_name] = predicted
    
# Convert to list of rows for DataFrame
rows = list(qa_dict.values())
df = pd.DataFrame(rows)

# Save as CSV or display
df.to_csv("merged_test_results.csv", index=False)
print(df.head())