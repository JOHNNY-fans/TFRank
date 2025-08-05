import os
import json
import math
import time
import traceback
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoConfig
from openai import OpenAI
import openai
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from call_vllm import call_vllm_api  # 确保 call_vllm.py 在同目录下


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='模型路径')
    parser.add_argument('--api_key', type=str, default='any-string', help='API KEY')
    parser.add_argument('--api_base', type=str, default='http://localhost:8113/v1', help='API BASE')
    parser.add_argument('--input_dir', type=str, required=True, help='输入jsonl目录')
    parser.add_argument('--result_dir', type=str, default='../result', help='结果根目录')
    parser.add_argument('--max_new_tokens_init', type=int, default=1024, help='初始生成token数')
    parser.add_argument('--workers', type=int, default=32, help='并发数')
    parser.add_argument('--think_mode', type=str2bool, default=False)
    parser.add_argument('--reasoning_model', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度')
    return parser.parse_args()

def get_model_short_name(model_path):
    # 如果最后一段是checkpoint-xxx，则取倒数第二段，否则取最后一段
    parts = model_path.rstrip('/').split(os.sep)
    if parts[-1].startswith('checkpoint-'):
        return parts[-3]
    else:
        return parts[-1]

def get_model_type(model_path):
    # 取倒数第三段（如 rearanker_msmarco_only_cot_sft-qwen3-0.6B）
    parts = model_path.rstrip('/').split(os.sep)
    if parts[-1].startswith('checkpoint-'):
        return parts[-3]
    else:
        return parts[-1]

def get_max_model_tokens(model_path):
    try:
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "max_position_embeddings"):
            return config.max_position_embeddings
        elif hasattr(config, "n_positions"):
            return config.n_positions
        elif hasattr(config, "seq_length"):
            return config.seq_length
        else:
            return 40960  # fallback
    except Exception as e:
        print(f"Warning: cannot load model config from {model_path}, use default 40960")
        return 40960

import math

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def get_logit(top_logprob_dict, candidates, fallback):
    for token in candidates:
        if token in top_logprob_dict:
            return top_logprob_dict[token]
    return fallback

def parse_yesno(top_logprob_dict):
    yes_candidates = ['yes', 'Yes', 'YES']
    no_candidates = ['no', 'No', 'NO']
    min_logprob = min(top_logprob_dict.values()) if top_logprob_dict else -100

    true_logit = get_logit(top_logprob_dict, yes_candidates, min_logprob - 10)
    false_logit = get_logit(top_logprob_dict, no_candidates, min_logprob - 10)

    scores = softmax([true_logit, false_logit])
    return {'yes': scores[0], 'no': scores[1]}

def parse_label(top_logprob_dict, target_label_tokens_str):
    min_logprob = min(top_logprob_dict.values()) if top_logprob_dict else -100
    logits = []
    for num_str in target_label_tokens_str:
        logp = top_logprob_dict.get(num_str, min_logprob-10)
        logits.append(logp)
    probs = softmax(logits)
    return {int(num): prob for num, prob in zip(target_label_tokens_str, probs)}

def get_next_token_probs(response_stream, tokenizer, think_mode=False, debug=False):
    buffer_tokens = []
    found_label_end = False
    found_yesno_end = False if think_mode else True
    label_finish = False
    yesno_finish = False
    label_result = None
    yesno_result = None
    target_label_tokens_str = ['0', '1', '2', '3', '4']
    for chunk in response_stream:
        if chunk[0] != "choices":
            continue
        choices = chunk[1]
        choice = choices[0]
        tokens = choice.logprobs.tokens
        token_logprobs = choice.logprobs.token_logprobs
        top_logprobs = choice.logprobs.top_logprobs
        for i, token in enumerate(tokens):
            buffer_tokens.append(token)
            text_so_far = tokenizer.convert_tokens_to_string(buffer_tokens)
            if found_yesno_end and not yesno_finish:
                top_logprob_dict = top_logprobs[i] if top_logprobs and i < len(top_logprobs) else {}
                yesno_result = parse_yesno(top_logprob_dict)
                yesno_finish = True
            if found_label_end and not label_finish:
                top_logprob_dict = top_logprobs[i] if top_logprobs and i < len(top_logprobs) else {}
                label_result = parse_label(top_logprob_dict, target_label_tokens_str)
                label_finish = True
            if debug:
                print(text_so_far)
            if "</think>\n\n" in text_so_far:
                found_yesno_end = True
            if "yes(" in text_so_far or "no(" in text_so_far:
                found_label_end = True
            if "yes (" in text_so_far or "no (" in text_so_far:
                found_label_end = True
            if "Yes(" in text_so_far or "No(" in text_so_far:
                found_label_end = True
            if "Yes (" in text_so_far or "No (" in text_so_far:
                found_label_end = True
            # if "Yes" in text_so_far or "No" in text_so_far:
            #     found_label_end = True
            if label_result and yesno_result:
                return label_result, yesno_result, text_so_far
    return None

def build_completion_prompt(
    input_text: str,
    system_prompt: str,
    tokenizer,
    max_model_tokens: int = 40960,
    reserved_tokens: int = 1024,
    think_mode: bool = True,
    reasoning_model: bool = True,
    try_num: int = 2
) -> str:
    if think_mode:
        fixed_prompt = (
            "<|im_start|>system\n"
            + system_prompt + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
    else:
        if reasoning_model:
            fixed_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "\n<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
            fixed_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "\n<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
    fixed_tokens = len(tokenizer.encode(fixed_prompt))
    max_input_tokens = max_model_tokens - reserved_tokens - fixed_tokens
    input_tokens = tokenizer.encode(input_text)
    if len(input_tokens) > max_input_tokens:
        input_tokens = input_tokens[:max_input_tokens]
    truncated_input = tokenizer.decode(input_tokens)
    if try_num >= 1:
        truncated_input += '\nPlease directly output the relevance judgment (yes or no), followed by the relevance score in parentheses, e.g., yes(score) or no(score).'
    if reasoning_model:
        prompt = truncated_input + (' /think' if think_mode else ' /no think')
    else:
        prompt = truncated_input
    if think_mode:
        completion_prompt = (
            "<|im_start|>system\n"
            + system_prompt + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + prompt + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
    else:
        if reasoning_model:
            completion_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
            completion_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
    return completion_prompt, len(tokenizer.encode(completion_prompt))

def deal(item, model_name, file_path, tokenizer, system_prompt, client, max_model_tokens, max_new_tokens_init, temperature=0., think_mode=True, reasoning_model=True, debug=False):
    max_try_retry = 1 if debug else 8
    max_tokens = max_model_tokens
    max_new_tokens = max_new_tokens_init
    try_num = 0
    while(True):
        try:
            completion_prompt, prompt_token = build_completion_prompt(
                input_text=item['input'],
                system_prompt=system_prompt,
                tokenizer=tokenizer,
                max_model_tokens=max_tokens,
                reserved_tokens=max_new_tokens,
                think_mode=think_mode,
                reasoning_model=reasoning_model,
                try_num=try_num,
            )
            response_stream = call_vllm_api(
                client,
                task="completion",
                model='rele_pointwise',
                prompt=completion_prompt,
                max_tokens=min(max_tokens - prompt_token, max_new_tokens),
                temperature=temperature,
                extra_body={"logprobs": 20}
            )
            response_chunks = list(response_stream)
            buffer_tokens = []
            for chunk in response_chunks:
                if chunk[0] != "choices":
                    continue
                choices = chunk[1]
                choice = choices[0]
                tokens = choice.logprobs.tokens
                if tokens:
                    buffer_tokens.append(tokens[0])
                    text_so_far = tokenizer.convert_tokens_to_string(buffer_tokens)
                break
            tmp_res = get_next_token_probs(response_chunks, tokenizer, think_mode=think_mode, debug=debug)
            if tmp_res is None and try_num!= max_try_retry:
                try_num +=1
                continue
            if try_num==max_try_retry and tmp_res is not None:
                label_logits_result, yesno_logits_result, response = tmp_res
                if debug:
                    print(response)
                item['predicted_label_max'] = max(label_logits_result, key=label_logits_result.get)
                weighted_score = sum([i * p for i, p in label_logits_result.items()])
                item['predicted_label_avg'] = weighted_score
                item['predicted_yesno'] = max(yesno_logits_result, key=yesno_logits_result.get)
                item['predicted_yesno_score'] = yesno_logits_result.get('yes', 0)
                item['response'] = response + ')'
            elif try_num==max_try_retry:
                item['predicted_label_max'] = 'error'
                item['predicted_label_avg'] = 0.5
                item['predicted_yesno'] = 'error'
                item['predicted_yesno_score'] = 0.5
                item['response'] = 'error'
            else:
                try_num+=1
                continue
            if not debug:
                with open(file_path, "a+", encoding="utf8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                max_tokens = max_new_tokens_init
                max_new_tokens = max_new_tokens_init
                break
            try_num+=1
        except Exception as e:
            str_e = str(e)
            if 'InvalidRequestError' in str_e:
                if 'maximum context' in str_e:
                    max_new_tokens += 1024
                continue
            elif 'JSONDecodeError' in str_e:
                pass
            else:
                traceback.print_exc()

def list_jsonl_files(dir_path):
    p = Path(dir_path)
    return [str(f.resolve()) for f in p.glob("*.jsonl")]

def main():
    args = parse_args()
    # 解析模型名和类型
    model_short_name = get_model_short_name(args.model_name)
    model_type = get_model_type(args.model_name)
    # 解析输入数据集名
    input_dir_last = os.path.basename(os.path.normpath(args.input_dir))
    # 结果目录
    result_dir = os.path.join(args.result_dir, input_dir_last, model_type)
    think_type = 'think' if args.think_mode else 'no_think'
    os.makedirs(os.path.join(result_dir, think_type), exist_ok=True)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 获取max_model_tokens
    max_model_tokens = get_max_model_tokens(args.model_name)
    # openai client
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    print(args.model_name)
    print(args.api_base)
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    system_prompt = 'Based on the relevance of the Documents to the Query and the Instruct provided to complete the task.'
    jsonl_list = list_jsonl_files(args.input_dir)
    for input_file_name in jsonl_list:
        with open(input_file_name, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        base_name = os.path.basename(input_file_name)
        dataset_name = os.path.splitext(base_name)[0].split('.')[0]
        with open(input_file_name, 'r', encoding='utf-8') as f:
            L = f.readlines()
        input_data = [json.loads(i) for i in L]
        task_type = input_data[0]['task_type']
        file_path = os.path.join(result_dir, think_type, f"{dataset_name}.{model_short_name}_{task_type}_{think_type}.jsonl")
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass
        with open(file_path, 'r', encoding='utf-8') as f:
            L = f.readlines()
        if len(L) > 0:
            finish_id = {str(json.loads(i)['qid']) + '#' + str(json.loads(i)['docid']) for i in L}
        else:
            finish_id = set()
        input_list = [i for i in input_data if str(i['qid']) + '#' + str(i['docid']) not in finish_id]
        print(dataset_name)
        print(len(input_list))
        print('***************')
        if len(input_list) == 0:
            continue
        if args.debug:
            print(args)
            deal(input_list[0], model_short_name, file_path, tokenizer, system_prompt, client,
                max_model_tokens, args.max_new_tokens_init, args.temperature, args.think_mode, args.reasoning_model, args.debug)
            break
        else:
            with tqdm_joblib(desc="My calculation", total=len(input_list)):
                Parallel(n_jobs=args.workers, prefer="threads")(
                    [delayed(deal)(
                        x, model_short_name, file_path, tokenizer, system_prompt, client,
                        max_model_tokens, args.max_new_tokens_init, args.temperature, args.think_mode, args.reasoning_model, args.debug
                    ) for x in input_list]
                )

if __name__ == "__main__":
    main()