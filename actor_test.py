from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm
from datasets import load_dataset
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
TOTAL_NUM_SAMPLES = 1000
INPUT_LEN = 64
ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True,token='')
ds_iterator = iter(ds.take(TOTAL_NUM_SAMPLES))
max_diffs = {}
for _ in tqdm(range(TOTAL_NUM_SAMPLES)):
    next_data = next(ds_iterator)["content"]
    all_input_ids = tokenizer(
        [next_data], return_tensors="pt", max_length=INPUT_LEN, truncation=True
    ).input_ids.to(model.device)

    # process the whole sequence
    all_outputs = model(all_input_ids, output_hidden_states=True, return_dict=True)
    # get logits for the last token
    last_token_logits = all_outputs.logits[0][-1:]

    # process the sequence except the last token
    kv = model(all_input_ids[:, :-1]).past_key_values
    # input only the last token with previous kv_cache
    new_output = model(all_input_ids[:, -1:], past_key_values=kv, output_hidden_states=True, return_dict=True)
    # extract the last token logits
    new_last_token_logits = new_output.logits[0][-1:]

    for layer_idx in range(len(all_outputs.hidden_states)):
        max_diff = torch.abs(
            all_outputs.hidden_states[layer_idx][:, -1, :] - new_output.hidden_states[layer_idx]
        ).max()
        max_diffs.setdefault(f"layer {layer_idx}", []).append(max_diff.cpu().item())
    # theese two distributions should be equal, but they are not.
    max_diffs.setdefault("logits", []).append(torch.abs(last_token_logits - new_last_token_logits).max().cpu().item())

print(max_diffs)

prompt1 = "Hey, are you conscious? Can you talk to me?"
#prompt2 = "Give me a recipe for porridge."
prompt2 = "How is the weather today?"

batch_1 = tokenizer.batch_encode_plus(
    [prompt1, prompt2], padding=True, return_tensors="pt",padding_side='left'
).to(model.device)
batch_2 = tokenizer.batch_encode_plus(
    [prompt1, prompt2], padding=True, return_tensors="pt",padding_side='right'
).to(model.device)

position_ids_1 = (batch_1.attention_mask.cumsum(dim=1) - 1) * batch_1.attention_mask
position_ids_2 = (batch_2.attention_mask.cumsum(dim=1) - 1) * batch_2.attention_mask
# Generate
generate_ids_1 = model(
    batch_1.input_ids, attention_mask=batch_1.attention_mask, position_ids=position_ids_1, 
    max_length=50)
generate_ids_2 = model(
    batch_2.input_ids, attention_mask=batch_2.attention_mask, position_ids=position_ids_2,
    max_length=50)

(generate_ids_1.logits[0,-1] - generate_ids_2.logits[0,-1]).max()
#############

ss1 = "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>"

ss2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"

ss1_idx = tokenizer(ss1,return_tensors="pt")
ss2_idx = tokenizer(ss2,return_tensors="pt")

#ss2_idx['input_ids'][0,:118] = 0

ss_idx = torch.zeros(2,127,dtype=torch.int64,device="cuda")
attention_mask = torch.zeros_like(ss_idx)

ss_idx[0,:ss1_idx['input_ids'].size(1)] = ss1_idx['input_ids']
attention_mask[0,4:ss1_idx['input_ids'].size(1)] = 1

ss_idx[1,-ss2_idx['input_ids'].size(1):] = ss2_idx['input_ids']
attention_mask[1,-ss2_idx['input_ids'].size(1):] = 1

position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)

output = model(ss_idx, attention_mask=attention_mask, position_ids=position_ids)
output["logits"] = output["logits"].to(torch.float32)


logits = output["logits"][:, :-1, :]
labels = ss_idx[:, 1:]

logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
logsumexp_values = torch.stack(
    [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
)
log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)

log_probs_labels[0,4:13] - log_probs_labels[1,-9:]