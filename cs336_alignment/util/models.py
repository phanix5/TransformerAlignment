import torch
from transformers import PreTrainedModel

from cs336_alignment.util.tokenizing import compute_entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    # logits should be batch_size seq_len vocab_size
    logits_log_softmax = torch.log_softmax(logits, dim=-1)
    # labels should be batch_size, seq_len
    log_probs = logits_log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    token_entropy = None
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
    
    return {
        "log_probs": log_probs,
        "token_entropy": token_entropy
    }

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = policy_log_probs.size(0)
    seq_len = policy_log_probs.size(1)
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant) / (batch_size * gradient_accumulation_steps)
    loss.backward()
    return loss, {}
    

