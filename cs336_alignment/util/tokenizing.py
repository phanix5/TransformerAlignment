from transformers import PreTrainedTokenizer
import torch

def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).

    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            a mask on the response tokens in the labels.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = 0
    else:
        pad_token_id = tokenizer.pad_token_id

    full_ids_list = []
    response_mask_list = []

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt_str)
        output_ids = tokenizer.encode(output_str)
        full_ids = prompt_ids + output_ids
        
        # Mask: 0 for prompt, 1 for output
        # We create a mask for the full sequence first.
        full_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        
        full_ids_list.append(full_ids)
        response_mask_list.append(full_mask)

    # Find max length for padding
    # If the list is empty, we should handle it, but assuming valid input.
    if not full_ids_list:
        return {
            "input_ids": torch.empty(0, 0, dtype=torch.long),
            "labels": torch.empty(0, 0, dtype=torch.long),
            "response_mask": torch.empty(0, 0, dtype=torch.long) # or bool
        }

    max_len = max(len(ids) for ids in full_ids_list)

    padded_input_ids = []
    padded_masks = []

    for ids, mask in zip(full_ids_list, response_mask_list):
        pad_len = max_len - len(ids)
        
        # Pad ids
        padded_ids = ids + [pad_token_id] * pad_len
        
        # Pad mask (padding is not part of response)
        padded_mask = mask + [0] * pad_len
        
        padded_input_ids.append(padded_ids)
        padded_masks.append(padded_mask)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, dtype=torch.long)

    # Slice to create final tensors
    # input_ids: remove the last token
    final_input_ids = input_ids_tensor[:, :-1]
    
    # labels: remove the first token (shifted)
    final_labels = input_ids_tensor[:, 1:]
    
    final_response_mask = mask_tensor[:, 1:]

    return {
        "input_ids": final_input_ids,
        "labels": final_labels,
        "response_mask": final_response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.
    """
    softmax = torch.nn.Softmax(dim=-1)
    return -1*torch.sum(softmax(logits)*torch.log_softmax(logits, dim=-1), dim=-1)


