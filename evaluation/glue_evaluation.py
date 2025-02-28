import random

import numpy as np
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm


def evaluate_glue(model_object, tasks=None, samples_per_task=None, seed=42):
    # Set seed
    random.seed(seed)

    # Iterate through all tasks if none provided
    if tasks is None:
        tasks = [
            "cola",
            "mnli",
            # "mrpc",
            "qnli",
            # "qqp",
            "rte",
            "stsb",
            "wnli",
            "sst2",
        ]

    # Prepare results dictionary
    all_results = {}

    # Iterate through selected tasks
    for task in tasks:
        print(
            f"\nEvaluating {task.upper()} with {samples_per_task if samples_per_task is not None else "all"} sample(s)..."
        )

        # Load dataset and metric
        dataset = load_dataset("glue", task)
        metric = load("glue", task)

        # Get validation split(s)
        splits = (
            ["validation_matched", "validation_mismatched"]
            if task == "mnli"
            else ["validation"]
        )
        task_results = {}

        for split in splits:
            results = []
            labels = []
            val_data = dataset[split]

            # Determine the number of samples to use
            total_examples = len(val_data)
            samples_to_use = (
                samples_per_task if samples_per_task is not None else total_examples
            )

            # Sample indices
            sample_indices = random.sample(
                range(total_examples), min(samples_to_use, total_examples)
            )

            for idx in tqdm(sample_indices):
                example = val_data[idx]

                # Prepare prompt based on task
                if task in ["mrpc", "wnli"]:
                    prompt = f"Are these sentences equivalent?\nSentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nAnswer with 0 or 1."
                elif task == "sst2":
                    prompt = f"Sentiment analysis: {example['sentence']}\nAnswer 0 for negative, 1 for positive."
                elif task == "cola":
                    prompt = f"Is this grammatically correct: {example['sentence']}\nAnswer 0 for no, 1 for yes."
                elif task == "qqp":
                    prompt = f"Are these questions equivalent?\nQ1: {example['question1']}\nQ2: {example['question2']}\nAnswer 0 or 1."
                elif task == "mnli":
                    prompt = (
                        f"Determine if the premise entails, contradicts, or is neutral to the hypothesis.\n"
                        f"Premise: {example['premise']}\n"
                        f"Hypothesis: {example['hypothesis']}\n"
                        f"Answer with: 0 for contradiction, 1 for entailment, 2 for neutral"
                    )
                elif task == "rte":
                    prompt = f"Does the premise entail the hypothesis?\nPremise: {example['sentence1']}\nHypothesis: {example['sentence2']}\nAnswer 0 for no, 1 for yes."
                elif task == "qnli":
                    prompt = f"Does the sentence answer the question?\nQuestion: {example['question']}\nSentence: {example['sentence']}\nAnswer 0 for no, 1 for yes."

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Respond only with the number corresponding to the correct answer.",
                    },
                    {"role": "user", "content": prompt},
                ]

                try:
                    output = model_object.chat_completion(
                        dialogs=[messages],
                        max_gen_len=10,
                    )
                    response = output[0]["generation"]["content"].strip()

                    # Extract first integer from response
                    filtered_response = "".join(filter(str.isdigit, response))

                    if filtered_response == "":
                        pred = -1
                    else:
                        pred = int(filtered_response)

                    results.append(pred)
                    labels.append(example["label"])
                except Exception as e:
                    print(f"Error processing example {idx}: {str(e)}")
                    continue

            if results:
                metrics = metric.compute(
                    predictions=np.array(results),
                    references=np.array(labels),
                )

                task_results[split] = {
                    "metrics": metrics,
                    "num_samples": len(results),
                }
            else:
                task_results[split] = {
                    "metrics": "Failed to process any examples",
                    "num_samples": 0,
                }

        all_results[task] = (
            task_results if len(splits) > 1 else task_results["validation"]
        )

    return all_results
