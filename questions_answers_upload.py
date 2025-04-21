from datasets import load_dataset, DatasetDict, Dataset
import json

config_to_filename_mapping = {
    "v1": "questions_answers.jsonl",
    "v2": "questions_answers_v2.jsonl",
}

config_to_validation_test_splits = {
    "v1": "validation_test_splits_v1.jsonl",
    "v2": "validation_test_splits_v2.jsonl",
}

def get_all_keys(*datasets):
    keys = set()
    for ds in datasets:
        for example in ds:
            keys.update(example.keys())
    return list(keys)

def pad_dataset(ds: Dataset, all_keys):
    def pad(example):
        return {k: example.get(k, None) for k in all_keys}
    return ds.map(pad)


for config, valtest_filename in config_to_validation_test_splits.items():
    train_ds = load_dataset("json", data_files=config_to_filename_mapping[config], split="train").shuffle(seed=0)
    train_features = train_ds.features
    
    # Load and split validation/test
    dataset = load_dataset("json", data_files=valtest_filename, split="train").shuffle(seed=0)
    split_dataset = dataset.train_test_split(test_size=0.5)
    
    # Get all possible keys across all splits
    all_keys = get_all_keys(train_ds, split_dataset["train"], split_dataset["test"])
    
    # Pad each dataset to have same keys
    train_ds = pad_dataset(train_ds, all_keys)
    val_ds = pad_dataset(split_dataset["train"], all_keys)
    test_ds = pad_dataset(split_dataset["test"], all_keys)
    
    # Assemble dataset dict and push
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds.cast(train_features),
        "test": test_ds.cast(train_features),
    })

    dataset_dict.push_to_hub("keeve101/sutd-qa-dataset", config)
