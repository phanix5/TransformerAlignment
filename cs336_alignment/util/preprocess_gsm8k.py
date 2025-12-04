import json
import os
import re
from datasets import load_dataset

def get_r1_zero_prompt_template():
    """
    Loads the r1_zero prompt template and removes the trailing <think> tag
    to prepare it for SFT (where <think> starts the response).
    """
    # This script is in cs336_alignment/util/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to cs336_alignment/
    base_dir = os.path.dirname(current_dir)
    prompt_path = os.path.join(base_dir, "prompts", "r1_zero.prompt")
    
    with open(prompt_path, "r") as f:
        template = f.read()
        
    return template

def clean_reasoning(text):
    """
    Removes tool call identifiers <<...>> from the text.
    """
    return re.sub(r"<<.*?>>", "", text)

def process_example(example, template):
    question = example["question"]
    raw_answer = example["answer"]
    
    # Split reasoning and answer
    # GSM8K format: [Reasoning] #### [Answer]
    if "####" not in raw_answer:
        return None
        
    reasoning_part, answer_part = raw_answer.split("####", 1)
    
    reasoning = clean_reasoning(reasoning_part).strip()
    answer = answer_part.strip()
    
    # Construct the prompt
    prompt = template.format(question=question)
    
    # Construct the response with <think> and <answer> tags
    # Note: prompt ends with <think>, so response starts with content
    response = f"{reasoning}</think> <answer>{answer}</answer>"
    
    return {
        "prompt": prompt,
        "response": response
    }

def main():
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main")
    
    template = get_r1_zero_prompt_template()
    
    # Determine output directory (data/gsm8k relative to project root)
    # cs336_alignment/util/ -> cs336_alignment/ -> project_root/ -> data/gsm8k/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    output_dir = os.path.join(project_root, "data", "gsm8k")
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ds.keys():
        output_path = os.path.join(output_dir, f"{split}_sft.jsonl")
        print(f"Processing {split} split -> {output_path}")
        
        with open(output_path, "w") as f:
            for example in ds[split]:
                processed = process_example(example, template)
                if processed:
                    f.write(json.dumps(processed) + "\n")
                    
    print("Done.")

if __name__ == "__main__":
    main()

