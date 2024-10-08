import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from collections import deque
import os

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SEQUENCES_FILE = "sequences.txt"


class Node:
    def __init__(self, token_id, parent=None):
        self.token_id = token_id
        self.children = []
        self.parent = parent
        self.is_branch_point = False
        self.probability = None
        self.entropy = None


def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    # Autodetect device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    return tokenizer, model, device


def calculate_entropy(logits):
    probs = softmax(logits.detach().cpu().numpy())
    return entropy(probs)


def generate_with_branching(prompt, tokenizer, model, device, max_length=50, probability_threshold=0.20,
                            temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    root = Node(input_ids[0][-1].item())
    stack = deque([(root, input_ids, 0)])
    sequence_counter = 0
    last_branch_point = 0

    with torch.no_grad():
        while stack:
            current_node, current_ids, step = stack.pop()

            if step >= max_length:
                continue

            outputs = model(input_ids=current_ids)
            next_token_logits = outputs.logits[0, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)

            entropy_value = calculate_entropy(next_token_logits)
            current_node.entropy = entropy_value

            print_step_info(step, current_ids, next_token_logits, tokenizer, prompt, last_branch_point)

            valid_continuations = get_valid_continuations(probabilities, probability_threshold)

            if len(valid_continuations) >= 2:
                last_branch_point = step
                process_branch_point(current_node, valid_continuations, current_ids, stack, step, tokenizer, device)
            else:
                sequence_counter = process_linear_continuation(current_node, valid_continuations, next_token_logits, probabilities,
                                            current_ids, stack, step, tokenizer, device, sequence_counter)

    return root


def print_step_info(step, current_ids, next_token_logits, tokenizer, prompt, last_branch_point):
    print(f"\nStep {step + 1}.")

    prompt_tokens = tokenizer.encode(prompt)
    prompt_end = len(prompt_tokens)

    current_tokens = current_ids[0].cpu().tolist()

    prompt_text = tokenizer.decode(current_tokens[:prompt_end])
    generated_tokens = current_tokens[prompt_end:]

    if last_branch_point > 0:
        branch_start = prompt_end + last_branch_point
        generated_text = tokenizer.decode(generated_tokens[:last_branch_point])
        last_branch_text = tokenizer.decode(generated_tokens[last_branch_point:])
    else:
        generated_text = tokenizer.decode(generated_tokens)
        last_branch_text = ""

    colored_text = f"{BLUE}{prompt_text}{RESET}{GREEN}{generated_text}{RESET}{MAGENTA}{last_branch_text}{RESET}"
    print(f"Text = {colored_text}")
    print(f"Full distribution entropy: {calculate_entropy(next_token_logits):.4f}")


def get_valid_continuations(probabilities, probability_threshold):
    return [(token_id, prob.item()) for token_id, prob in enumerate(probabilities) if
            prob.item() > probability_threshold]


def process_branch_point(current_node, valid_continuations, current_ids, stack, step, tokenizer, device):
    current_node.is_branch_point = True

    # Sort valid continuations by probability in descending order
    sorted_continuations = sorted(valid_continuations, key=lambda x: x[1], reverse=True)

    for token_id, token_prob in sorted_continuations:
        new_node = Node(token_id, parent=current_node)
        new_node.probability = token_prob
        current_node.children.append(new_node)
        new_ids = torch.cat([current_ids, torch.tensor([[token_id]], device=device)], dim=1)

        # Append to the left side of the deque to prioritize exploration
        stack.appendleft((new_node, new_ids, step + 1))
        print(
            f"Branching: {YELLOW}{tokenizer.decode([token_id])}{RESET}, probability = {token_prob:.4f}, total branches = {len(stack)}")


def process_linear_continuation(current_node, valid_continuations, next_token_logits, probabilities, current_ids, stack,
                                step, tokenizer, device, sequence_counter):
    if valid_continuations:
        token_id, token_prob = valid_continuations[0]
    else:
        token_id = torch.argmax(next_token_logits).item()
        token_prob = probabilities[token_id].item()

    new_node = Node(token_id, parent=current_node)
    new_node.probability = token_prob
    current_node.children.append(new_node)
    new_ids = torch.cat([current_ids, torch.tensor([[token_id]], device=device)], dim=1)

    if token_id == tokenizer.eos_token_id:
        print("End of text token reached.")
        sequence_counter += 1
        log_sequence(new_node, count_branch_points(new_node), sequence_counter, tokenizer)
    else:
        stack.append((new_node, new_ids, step + 1))
        print(f"Linear: {CYAN}{tokenizer.decode([token_id])}{RESET}, probability = {token_prob:.4f}")

    return sequence_counter


def count_branch_points(node):
    count = 0
    current = node
    while current.parent is not None:
        if current.parent.is_branch_point:
            count += 1
        current = current.parent
    return count


def log_sequence(end_node, branch_points, sequence_number, tokenizer):
    sequence = []
    probabilities = []
    entropies = []
    current = end_node
    while current is not None:
        sequence.insert(0, current.token_id)
        if current.probability is not None:
            probabilities.insert(0, current.probability)
        if current.entropy is not None and current.is_branch_point:
            entropies.insert(0, current.entropy)
        current = current.parent

    probability_product = np.prod(probabilities)
    entropy_sum = sum(entropies)

    with open(SEQUENCES_FILE, "a") as f:
        f.write(f"Sequence: {sequence_number}\n")
        f.write(f"Branch Points: {branch_points}\n")
        f.write(f"Probabilities at branch points: {', '.join(['%.02f' % p for p in probabilities])}\n")
        f.write(f"Product of probabilities: {probability_product:.05f}\n")
        f.write(f"Entropies at branch points: {', '.join(['%.02f' % e for e in entropies])}\n")
        f.write(f"Sum of entropies: {entropy_sum:.02f}, avg. is {entropy_sum/(len(entropies)-0.0000000001):.02f}\n")
        f.write(f"Text: {tokenizer.decode(sequence)}\n")
        f.write("-" * 50 + "\n")


def print_tree(node, tokenizer, depth=0, prefix=""):
    if node.parent is None:
        print(f"{prefix}Root: {tokenizer.decode([node.token_id])}")
    else:
        print(f"{prefix}{'[B] ' if node.is_branch_point else ''}{tokenizer.decode([node.token_id])}")

    for i, child in enumerate(node.children):
        new_prefix = prefix + ("└── " if i == len(node.children) - 1 else "├── ")
        print_tree(child, tokenizer, depth + 1, new_prefix)


def main():
    if os.path.exists(SEQUENCES_FILE):
        os.remove(SEQUENCES_FILE)

    tokenizer, model, device = setup_model()

    prompt = """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant with advanced analytical capabilities.
Carefully analyze user input.
Take a step back to reflect on the nature of request, and think step by step to provide a response.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What number is larger 9.11 or 9.9?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    result_tree = generate_with_branching(prompt, tokenizer, model, device, max_length=1024, probability_threshold=0.30,
                                          temperature=0.7)

    print("\nGenerated Tree Structure:")
    print_tree(result_tree, tokenizer)


if __name__ == '__main__':
    main()
