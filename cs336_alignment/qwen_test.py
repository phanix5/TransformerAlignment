import json
import os
from typing import Callable, List, Dict
from vllm import LLM, SamplingParams
from datasets import load_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # Generate outputs
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    # Iterate over results and ground truths
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        # Compute reward
        metrics = reward_fn(generated_text, ground_truth)
        
        # Store result
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "metrics": metrics
        })
        
    # Serialize to disk
    output_path = "qwen_test_results.jsonl"
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
            
    print(f"Evaluation complete. Results saved to {output_path}")

def main():
    # Load the MATH data set
    ds = load_dataset("hiyouga/math12k")
    
    # Use the test set
    if "test" in ds:
        data = ds["test"]
    elif "validation" in ds:
        data = ds["validation"]
    else:
        print("Warning: 'test' or 'validation' split not found, using 'train'")
        data = ds["train"]
        
    # Load prompt template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "r1_zero.prompt")
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
        
    prompts = []
    ground_truths = []
    
    for example in data:
        # Extract question and solution
        # hiyouga/math12k usually follows 'problem' and 'solution' or similar
        question = example.get("problem") or example.get("question")
        solution = example.get("solution") or example.get("answer")
        
        if question and solution:
            formatted_prompt = prompt_template.format(question=question)
            prompts.append(formatted_prompt)
            ground_truths.append(solution)
            
    # Initialize vLLM
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # Evaluate
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths
    )

if __name__ == "__main__":
    main()
