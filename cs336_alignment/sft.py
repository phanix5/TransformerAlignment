import os
import json
import argparse
from typing import List, Callable, Dict, Any, Optional
from unittest.mock import patch
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset

from cs336_alignment.util.models import get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.util.tokenizing import tokenize_prompt_and_output
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def extract_xml_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return text


def log_generations(
    llm: LLM,
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    sampling_params: SamplingParams = None,
):
    """
    Prompt the model to generate responses and log statistics.
    """
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=1, top_p=1, max_tokens=1024, stop=["</answer>"],
            include_stop_str_in_output=True
        )

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    generated_responses = [output.outputs[0].text for output in outputs]

    # Tokenize prompts and responses for entropy calculation
    tokenized_data = tokenize_prompt_and_output(
        prompts, generated_responses, tokenizer
    )
    input_ids = tokenized_data["input_ids"].to(policy.device)
    labels = tokenized_data["labels"].to(policy.device)
    response_mask = tokenized_data["response_mask"].to(policy.device)

    # Calculate entropy and log probs
    with torch.no_grad():
        log_probs_dict = get_response_log_probs(
            policy, input_ids, labels, return_token_entropy=True
        )
    
    token_entropies = log_probs_dict["token_entropy"]

    total_response_len = 0
    total_correct_len = 0
    num_correct = 0
    total_incorrect_len = 0
    num_incorrect = 0

    for i, (prompt, response, ground_truth) in enumerate(zip(prompts, generated_responses, ground_truths)):
        # Calculate reward
        reward_info = reward_fn(response, ground_truth)
        
        # Calculate average token entropy for the response
        # Use response_mask to select only response tokens
        mask = response_mask[i] == 1
        if mask.sum() > 0:
            avg_token_entropy = token_entropies[i][mask].mean().item()
            response_length = mask.sum().item()
        else:
            avg_token_entropy = 0.0
            response_length = 0

        # Log example details
        print(f"Example {i+1}:")
        print(f"  Input Prompt: {prompt}")
        print(f"  Generated Response: {response}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Reward Info: {reward_info}")
        print(f"  Average Token Entropy: {avg_token_entropy:.4f}")
        print(f"  Response Length: {response_length}")
        print("-" * 50)

        # Update statistics
        total_response_len += response_length
        # Assuming reward_info contains "answer_reward" where 1.0 is correct
        is_correct = reward_info.get("answer_reward", 0.0) == 1.0
        
        if is_correct:
            total_correct_len += response_length
            num_correct += 1
        else:
            total_incorrect_len += response_length
            num_incorrect += 1

    # Log average statistics
    avg_response_len = total_response_len / len(prompts) if len(prompts) > 0 else 0.0
    avg_correct_len = total_correct_len / num_correct if num_correct > 0 else 0.0
    avg_incorrect_len = total_incorrect_len / num_incorrect if num_incorrect > 0 else 0.0

    print("Summary Statistics:")
    print(f"  Average Response Length: {avg_response_len:.2f}")
    print(f"  Average Response Length (Correct): {avg_correct_len:.2f}")
    print(f"  Average Response Length (Incorrect): {avg_incorrect_len:.2f}")
    accuracy = num_correct / len(prompts) if len(prompts) > 0 else 0.0
    print(f"  Accuracy: {accuracy:.2%}")


def load_gsm8k_data(filename: str, limit: Optional[int] = None):
    """
    Load GSM8K data from data/gsm8k directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, "data", "gsm8k")
    output_path = os.path.join(output_dir, filename)
    data = []
    with open(output_path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line)
            data.append((item["prompt"], item["response"]))
    return data


def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="SFT Training and Generation")
    parser.add_argument("--generate", action="store_true", help="Run generation instead of training")
    parser.add_argument("--prompt-indices", type=int, nargs="+", help="Indices of prompts to use from validation set")
    parser.add_argument("--dataset-size", type=int, help="Number of items to use for training")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size per optimizer step")
    parser.add_argument("--micro-batch-size", type=int, default=5, help="Microbatch size for gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", "cuda", 1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B") # load pretrained tokenizer from model
    preTrainedModel = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    load_policy_into_vllm_instance(preTrainedModel, llm)
    
    # Load prompt template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "r1_zero.prompt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    if args.generate:
        # Load the MATH data set
        ds = load_dataset("hiyouga/math12k")
        
        # Use the test set or validation set
        if "test" in ds:
            data = ds["test"]
        elif "validation" in ds:
            data = ds["validation"]
        else:
            print("Warning: 'test' or 'validation' split not found, using 'train'")
            data = ds["train"]

        # Select indices
        indices = args.prompt_indices if args.prompt_indices is not None else range(10)

        prompts = []
        ground_truths = []

        for idx in indices:
            if idx >= len(data):
                print(f"Warning: Index {idx} out of bounds, skipping.")
                continue
            
            example = data[idx]
            # Extract question and solution
            question = example.get("problem") or example.get("question")
            solution = example.get("solution") or example.get("answer")
            
            if question and solution:
                formatted_prompt = prompt_template.format(question=question)
                prompts.append(formatted_prompt)
                ground_truths.append(solution)
        
        log_generations(
            llm=llm,
            policy=preTrainedModel,
            tokenizer=tokenizer,
            prompts=prompts,
            ground_truths=ground_truths,
            reward_fn=r1_zero_reward_fn
        )
        return

    # prepare training data
    train_data = load_gsm8k_data("train_sft.jsonl", limit=args.dataset_size)
    tokenized_train_data = tokenize_prompt_and_output([x[0] for x in train_data], [x[1] for x in train_data], tokenizer)
    input_data_all = tokenized_train_data["input_ids"]
    labels_all = tokenized_train_data["labels"]
    response_mask_all = tokenized_train_data["response_mask"]

    optimizer = AdamW(preTrainedModel.parameters(), lr=args.lr)

    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size

    print(f"Starting training with dataset size: {len(train_data)}")

    preTrainedModel.train()
    num_batches = input_data_all.shape[0] // micro_batch_size
    for micro_batch in range(num_batches):
        print_gpu_memory(f"Start Micro-batch {micro_batch}")
        batch = input_data_all[micro_batch_size * micro_batch: micro_batch_size * (micro_batch + 1)]
        labels = labels_all[micro_batch_size * micro_batch: micro_batch_size * (micro_batch + 1)]
        response_mask = response_mask_all[micro_batch_size * micro_batch: micro_batch_size * (micro_batch + 1)]

        log_probs_dict = get_response_log_probs(preTrainedModel, batch, labels)
        print_gpu_memory(f"After Forward {micro_batch}")

        loss, _ = sft_microbatch_train_step(log_probs_dict["log_probs"], response_mask, batch_size // micro_batch_size)
        print_gpu_memory(f"After Backward {micro_batch}")
        
        if micro_batch % 10 == 0:
            print(f"Micro-batch {micro_batch}/{num_batches}, Loss: {loss.item()}")

        # Clip gradients with L2 norm cutoff of 1.0
        if ((micro_batch + 1) * micro_batch_size) % batch_size == 0:
            torch.nn.utils.clip_grad_norm_(preTrainedModel.parameters(), 1.0)
            print_gpu_memory(f"Before Step {micro_batch}")
            optimizer.step()
            print_gpu_memory(f"After Step {micro_batch}")
            optimizer.zero_grad()
            print_gpu_memory(f"After Zero Grad {micro_batch}")
            
    print("Training complete.")
    
    # Update vLLM weights for validation
    load_policy_into_vllm_instance(preTrainedModel, llm)
    
    # Validation on test_sft.jsonl
    print("\nEvaluating on test_sft.jsonl (first 100 examples)...")
    test_sft_data = load_gsm8k_data("test_sft.jsonl", limit=100)
    test_prompts = [x[0] for x in test_sft_data]
    test_responses = [x[1] for x in test_sft_data]
    test_ground_truths = [extract_xml_answer(r) for r in test_responses]
    
    log_generations(
        llm=llm,
        policy=preTrainedModel,
        tokenizer=tokenizer,
        prompts=test_prompts,
        ground_truths=test_ground_truths,
        reward_fn=r1_zero_reward_fn
    )
    
    # Validation on Math12K
    print("\nEvaluating on Math12K validation set (first 50 examples)...")
    ds = load_dataset("hiyouga/math12k")
    if "validation" in ds:
        val_data = ds["validation"]
    elif "test" in ds:
        val_data = ds["test"]
    else:
        val_data = ds["train"]
        
    val_indices = range(50)
    val_prompts = []
    val_ground_truths = []
    
    for idx in val_indices:
        if idx >= len(val_data):
            continue
        example = val_data[idx]
        question = example.get("problem") or example.get("question")
        solution = example.get("solution") or example.get("answer")
        
        if question and solution:
            formatted_prompt = prompt_template.format(question=question)
            val_prompts.append(formatted_prompt)
            val_ground_truths.append(solution)
            
    log_generations(
        llm=llm,
        policy=preTrainedModel,
        tokenizer=tokenizer,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        reward_fn=r1_zero_reward_fn
    )


if __name__ == "__main__":
    main()
    







    

    

