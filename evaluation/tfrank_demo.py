# evaluation/minimal_ranker.py

from typing import Tuple
from transformers import AutoTokenizer
from openai import OpenAI

from call_vllm import call_vllm_api
from run_eval import (
    build_completion_prompt,
    get_next_token_probs,
    get_max_model_tokens,
)


class TFRankDemoRanker:
    """
    Minimal demo wrapper for TFRank.

    Usage:
        ranker = TFRankDemoRanker(
            model_name="/path/to/your/checkpoint",
            api_base="http://localhost:8113/v1",
            api_key="any-string",
            think_mode=False,
            reasoning_model=False,
        )
        score, fg_score, yes_score, resp = ranker.score("query", "document")
    """

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8113/v1",
        api_key: str = "any-string",
        think_mode: bool = False,
        reasoning_model: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        system_prompt: str = (
            "Based on the relevance of the Documents to the Query and "
            "the Instruct provided to complete the task."
        ),
        model_id_in_vllm: str = "rele_pointwise",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_model_tokens = get_max_model_tokens(model_name)

        self.client = OpenAI(api_key=api_key, base_url=api_base)

        self.think_mode = think_mode
        self.reasoning_model = reasoning_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.model_id_in_vllm = model_id_in_vllm

    @staticmethod
    def _build_input(query: str, document: str) -> str:
        """
        构造与训练/评测一致的 input 文本。
        - 包含 0~4 评分解释
        - 包含 yes/no 规则
        - 末尾附带 /think 或 /no think
        """

        return (
            "<Instruct>: Please judge the relevance strength between the query and the document, "
            "and directly output the relevance judgment (yes or no), followed by the relevance "
            "score in parentheses, e.g., yes(score) or no(score).\n\n\n"
            "- Relevance scores are represented by numbers from 0 to 4, with the following meanings:\n"
            "0 means completely irrelevant,\n"
            "1 means weakly relevant,\n"
            "2 means moderately relevant,\n"
            "3 means strongly relevant,\n"
            "4 means completely relevant.\n\n"
            "- For binary relevance judgment (yes or no), the rule is:\n"
            "Scores 0 and 1 are considered irrelevant and represented as \"no\",\n"
            "Scores 2, 3, and 4 are considered relevant and represented as \"yes\".\n\n\n"
            f"<Query>: {query}\n\n\n"
            f"<Document>: {document}"
        )


    def score(self, query: str, document: str) -> Tuple[float, float, float, str, str]:
        """
        给定 query & doc，返回：
            final_score:   最终相关性分数，默认为细粒度(归一化)和 yes 概率的平均
            fg_score:      细粒度分数，归一化到 [0,1]
            yes_score:     yes 的概率 [0,1]
            response:      原始 LLM 文本输出

        细粒度原始标签是 0–4，我们这里先做期望，再 /4 归一化。
        """
        input_text = self._build_input(query, document)

        completion_prompt, prompt_token = build_completion_prompt(
            input_text=input_text,
            system_prompt=self.system_prompt,
            tokenizer=self.tokenizer,
            max_model_tokens=self.max_model_tokens,
            reserved_tokens=self.max_new_tokens,
            think_mode=self.think_mode,
            reasoning_model=self.reasoning_model,
            try_num=0,  # demo 不做多轮重试
        )

        response_stream = call_vllm_api(
            self.client,
            task="completion",
            model=self.model_id_in_vllm,
            prompt=completion_prompt,
            max_tokens=min(
                self.max_model_tokens - prompt_token,
                self.max_new_tokens,
            ),
            temperature=self.temperature,
            extra_body={"logprobs": 20},
        )
        response_chunks = list(response_stream)

        tmp_res = get_next_token_probs(
            response_chunks,
            tokenizer=self.tokenizer,
            think_mode=self.think_mode,
            debug=False,
        )
        if tmp_res is None:
            # 解析失败：全部给默认值
            return 0.5, 0.5, 0.5, completion_prompt, "[PARSE ERROR]"

        label_logits_result, yesno_logits_result, response = tmp_res

        # ---------- 细粒度分数（0–4 -> 0–1） ----------
        fg_score = None
        if label_logits_result:
            # 期望值：sum(label * prob)，label 是 0–4
            fg_raw = sum(float(k) * float(p) for k, p in label_logits_result.items())
            # 归一化到 [0,1]
            fg_score = max(0.0, min(fg_raw / 4.0, 1.0))

        # ---------- yes 概率 ----------
        yes_score = None
        if yesno_logits_result:
            yes_score = float(yesno_logits_result.get("yes", 0.0))
            yes_score = max(0.0, min(yes_score, 1.0))

        # ---------- 最终分数：两个分数的平均 ----------
        if fg_score is not None and yes_score is not None:
            final_score = (fg_score + yes_score) / 2.0
        elif fg_score is not None:
            final_score = fg_score
        elif yes_score is not None:
            final_score = yes_score
        else:
            final_score = 0.5

        return float(final_score), float(fg_score or 0.5), float(yes_score or 0.5), completion_prompt, response+')'
