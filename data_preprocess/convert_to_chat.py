from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def chatml_dpo_format(example):
    message = [
        { "role": "user","content": example['question']},
    ]
    prompt = tokenizer.apply_chat_template(message,
                                           tokenize=False,
                                           add_generation_prompt=True)
    chosen = example['chosen'] + "<end_of_turn>"
    rejected = example['rejected'] + "<end_of_turn>"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def convert_to_dpo_chat_template(dataset_name, model_name):
    orig_dataset = load_dataset(dataset_name)

    train_dataset = orig_dataset['train'].map(chatml_dpo_format,
                                              remove_columns=["question"])
    split_dataset = train_dataset.train_test_split(test_size=500, seed=42)
    dataset = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset['test']
    })
    dataset.push_to_hub(f'{dataset_name}-chat-gemma', private=True)


if __name__ == "__main__":
    dataset_name = "Translation-EnKo/nlp-arxiv-translation-dpo-with-math-10k"
    model_name = "HumanF-MarkrAI/Gukbap-Gemma2-9B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    convert_to_dpo_chat_template(dataset_name, model_name)