import asyncio
import re
import math
from typing import List, Tuple, Dict, Set
import numpy as np
import json

from swift.utils import get_logger

logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""

from swift.plugin import ORM, orms

class ThinkFormatReward(ORM):
    def __call__(self, completions, think_mode, **kwargs) -> list:
        rewards = []
        for content, tmode in zip(completions,think_mode):
            if tmode == "no_think":
                # <think>\n\n</think>\n\n 后面必须有非空答案内容
                pattern = r'<think>\n\n</think>\n\n(.+)'
                m = re.search(pattern, content, re.DOTALL)
                if m:
                    answer_part = m.group(1).strip()
                    reward = 1.0 if len(answer_part) > 0 else 0.0
                else:
                    reward = 0.0

            elif tmode == "think":
                # <think>\n思考过程\n</think>\n\n 后面必须有非空答案内容
                m = re.search(r'<think>\n(.*?)\n</think>\n\n(.+)', content, re.DOTALL)
                if m:
                    think_content = m.group(1).strip()
                    answer_part = m.group(2).strip()
                    token_count = len(think_content)
                    if 4096>= token_count >= 256 and len(answer_part) > 0:
                        reward = 1.0
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards

from typing import List, Tuple, Dict, Set

class ThinkRankReward(ORM):
    def __init__(self, max_score=4.0):
        self.max_score = max_score

    def __call__(self, completions: List[str], solution: List[str], task_type: List[str], 
                 predict_way: List[str], think_mode: List[str], **kwargs) -> List[float]:
        rewards = []
        for pred, true, ttype, pway, tmode in zip(completions, solution, task_type, predict_way, think_mode):
            if ttype == 'pointwise':
                reward = self._pointwise_reward(pred, true, pway, tmode)
            elif ttype == 'pairwise':
                reward = self._pairwise_reward(pred, true, pway, tmode)
            elif ttype == 'listwise':
                reward = self._listwise_reward(pred, true, pway, tmode)
            else:
                raise ValueError(f"Unsupported task_type: {ttype}")
            rewards.append(reward)
        return rewards

    # -------- pointwise --------

    def _pointwise_score_mse(self, pred_score, true_score):
        pred_score = float(pred_score)
        true_score = float(true_score)
        p = pred_score / self.max_score
        t = true_score / self.max_score
        mse = (p - t) ** 2
        return mse

    def _pointwise_reward(self, pred: str, true: str, predict_way: str, think_mode: str) -> float:
        """
        计算pointwise奖励，包含acc和mse两部分的平均
        """
        # 先提取answer部分（think部分之外）
        answer_pred = self._extract_answer_part(pred, think_mode)
        answer_true = true
        if answer_pred is None or answer_true is None:
            return 0.0

        if predict_way == "score_only":
            # answer应该是0,1,2,3,4中的一个整数
            if not self._is_valid_score(answer_pred) or not self._is_valid_score(answer_true):
                return 0.0
            acc = 1.0 if answer_pred == answer_true else 0.0
            mse = self._pointwise_score_mse(answer_pred, answer_true)
            mse_reward = max(0.0, 1.0 - math.sqrt(mse))
            return (acc + mse_reward) / 2

        elif predict_way == "yesno_only":
            # answer应该是yes或no
            if answer_pred.lower() not in ("yes", "no") or answer_true.lower() not in ("yes", "no"):
                return 0.0
            if answer_pred == answer_true:
                acc = 1.0
            elif answer_pred.lower() == answer_true.lower():
                acc = 0.5
            else:
                acc = 0.0
            return acc  # 只有acc，没有mse

        elif predict_way == "yesno_score":
            # answer格式是 yes(score) 或 no(score)
            m = re.match(r'^(yes|no)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)$', answer_pred, re.I)
            m_true = re.match(r'^(yes|no)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)$', answer_true, re.I)
            if not m or not m_true:
                return 0.0
            pred_label, pred_score = m.group(1).lower(), float(m.group(2))
            true_label, true_score = m_true.group(1).lower(), float(m_true.group(2))

            # 计算acc
            if pred_label == true_label:
                acc = 1.0
            elif pred_label.lower() == true_label.lower():
                acc = 0.5
            else:
                acc = 0.0

            # 计算mse
            mse = self._pointwise_score_mse(pred_score, true_score)
            mse_reward = max(0.0, 1.0 - math.sqrt(mse))

            # 检查分类、分数一致性
            if true_label.lower() == 'yes':
                if float(pred_score) < 2:
                    mse_reward = 0.
            if true_label.lower() == 'no':
                if float(pred_score) >= 2:
                    mse_reward = 0.
            if pred_label.lower() == 'yes':
                if float(pred_score) < 2:
                    mse_reward = 0.
            if pred_label.lower() == 'no':
                if float(pred_score) >= 2:
                    mse_reward = 0.

            return (acc + mse_reward) / 2

        else:
            # 不支持的predict_way
            return 0.0

    def _extract_answer_part(self, content: str, think_mode: str) -> str or None:
        # 复用之前严格格式提取
        if think_mode == "think":
            m = re.search(r'<think>\n(.*?)\n</think>\n\n(.+)', content, re.DOTALL)
            if m:
                answer = m.group(2).strip()
                return answer if answer else None
            return None
        elif think_mode == "no_think":
            m = re.search(r'<think>\n\n</think>\n\n(.+)', content, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                return answer if answer else None
            return None
        else:
            return None

    def _is_valid_score(self, s: str) -> bool:
        try:
            val = int(s)
            return val in (0,1,2,3,4)
        except:
            return False

    # ------------- pairwise -------------
    def _pairwise_reward(self, pred: str, true: str, predict_way: str, think_mode: str) -> float:
        # 提取answer部分
        answer_pred = self._extract_answer_part(pred, think_mode)
        answer_true = true
        if answer_pred is None or answer_true is None:
            return 0.0

        if predict_way == "id_only":
            # 解析关系对
            rel_pred = self._parse_pairwise_relation_id_only(answer_pred)
            rel_true = self._parse_pairwise_relation_id_only(answer_true)
            if rel_pred is None or rel_true is None:
                return 0.0
            # 计算acc
            acc = 1.0 if self._pairwise_relation_equal(rel_pred, rel_true) else 0.0
            return acc

        elif predict_way == "id_score":
            # 解析带分数的关系对
            rel_pred = self._parse_pairwise_relation_id_score(answer_pred)
            rel_true = self._parse_pairwise_relation_id_score(answer_true)
            if rel_pred is None or rel_true is None:
                return 0.0
            # 计算acc和mse
            acc, diff_acc = self._pairwise_id_score_acc(rel_pred, rel_true)
            mse = self._pairwise_id_score_mse(rel_pred, rel_true)
            mse_reward = max(0.0, 1.0 - math.sqrt(mse))
            return (acc + diff_acc + mse_reward) / 3

        else:
            return 0.0

    def _parse_pairwise_relation_id_only(self, s: str):
        """
        解析格式如 "[1] > [2]"、"[2] = [1]" 等，返回统一格式 (left_id, relation, right_id)
        relation in {">", "<", "="}
        """
        s = s.strip()
        m = re.match(r'\[(\d+)\]\s*([<>=])\s*\[(\d+)\]', s)
        if not m:
            return None
        left, rel, right = m.group(1), m.group(2), m.group(3)
        return (left, rel, right)

    def _pairwise_relation_equal(self, rel1, rel2) -> bool:
        """
        判断两个关系是否等价，考虑对称关系
        rel格式：(left_id, relation, right_id)
        """
        if rel1 == rel2:
            return True
        # 对称关系判断
        left1, r1, right1 = rel1
        left2, r2, right2 = rel2
        # 关系对称映射
        sym_map = {">": "<", "<": ">", "=": "="}
        if left1 == right2 and right1 == left2 and r1 == sym_map.get(r2, None):
            return True
        return False

    def _parse_pairwise_relation_id_score(self, s: str):
        """
        解析格式如 "[1](score1) > [2](score2)"，返回 (left_id, left_score, relation, right_id, right_score)
        """
        s = s.strip()
        m = re.match(r'\[(\d+)\]\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*([<>=])\s*\[(\d+)\]\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)', s)
        if not m:
            return None
        left_id, left_score, rel, right_id, right_score = m.group(1), float(m.group(2)), m.group(3), m.group(4), float(m.group(5))
        return (left_id, left_score, rel, right_id, right_score)

    def _pairwise_id_score_acc(self, pred, true) -> float:
        if pred is None or true is None:
            return 0.0
        p_left, p_score_left, p_rel, p_right, p_score_right = pred
        t_left, t_score_left, t_rel, t_right, t_score_right = true

        def diff_acc(diff):
            max_diff = 4
            return max(0.0, 1.0 - diff / max_diff)

        acc = 1.0 if self._pairwise_relation_equal(
            (p_left, p_rel, p_right),
            (t_left, t_rel, t_right)
        ) else 0.0

        orig_diff = abs(t_score_left - t_score_right)
        pred_diff = abs(p_score_left - p_score_right)
        diff = abs(pred_diff - orig_diff)

        return acc, diff_acc(diff)

    def _pairwise_id_score_mse(self, pred, true) -> float:
        """
        计算id_score的mse，分数部分
        """
        if pred is None or true is None:
            return 1.0  # mse最大
        p_left, p_score_left, p_rel, p_right, p_score_right = pred
        t_left, t_score_left, t_rel, t_right, t_score_right = true
        mse = (self._pointwise_score_mse(p_score_left,t_score_left) +  self._pointwise_score_mse(p_score_right,t_score_right)) / 2
        return mse

    def _inverse_relation(self, rel: str) -> str:
        inv_map = {">": "<", "<": ">", "=": "="}
        return inv_map.get(rel, rel)

    # -------- listwise --------

    def _listwise_reward(self, pred: str, true: str, predict_way: str, think_mode: str) -> float:
        def normalize(lst):
            return sorted([sorted(sub) for sub in lst])
        # 提取answer部分
        answer_pred = self._extract_answer_part(pred, think_mode)
        answer_true = true
        if answer_pred is None or answer_true is None:
            return 0.0

        if predict_way == "id_only":
            # 解析id_only格式
            order_pred = self._parse_listwise_id_only(answer_pred)
            order_true = self._parse_listwise_id_only(answer_true)
            if order_pred is None or order_true is None:
                return 0.0

            acc = 1.0 if normalize(order_pred) == normalize(order_true) else 0.0

            ndcg_score = self._listwise_ndcg(order_pred, order_true)
            docs_len = len([doc for group in order_true for doc in group])
            order_init = [[num+1 for num in range(docs_len)]]
            init_ndcg_score = self._listwise_ndcg(order_init, order_true)
            denom = 1 - init_ndcg_score
            if abs(denom) < 1e-8:
                ndcg_reward = ndcg_score - 1.0
            else:
                ndcg_reward = (ndcg_score - init_ndcg_score) / denom

            ndcg_reward = np.clip(ndcg_reward, -5.0, 5.0)

            return (acc + ndcg_reward) / 2.0

        elif predict_way == "id_score":
            # 解析id_score格式
            order_pred, scores_pred = self._parse_listwise_id_score(answer_pred)
            order_true, scores_true = self._parse_listwise_id_score(answer_true)
            if order_pred is None or order_true is None:
                return 0.0
            if set(sum(order_pred,[])) != set(sum(order_true,[])):
                return 0.0
            acc = 1.0 if normalize(order_pred) == normalize(order_true) else 0.0

            ndcg_score = self._listwise_ndcg(order_pred, order_true, scores_true)
            docs_len = len([doc for group in order_true for doc in group])
            order_init = [[num+1 for num in range(docs_len)]]
            init_ndcg_score = self._listwise_ndcg(order_init, order_true, scores_true)
            denom = 1 - init_ndcg_score
            if abs(denom) < 1e-8:
                ndcg_reward = ndcg_score - 1.0
            else:
                ndcg_reward = (ndcg_score - init_ndcg_score) / denom

            ndcg_reward = np.clip(ndcg_reward, -5.0, 5.0)

            mse = self._listwise_score_mse(scores_pred, scores_true)
            mse_reward = max(0.0, 1.0 - math.sqrt(mse))

            return (acc + ndcg_reward + mse_reward) / 3.0
        else:
            return 0.0

    def _extract_answer_part(self, content: str, think_mode: str) -> str or None:
        # 复用之前严格格式提取
        if think_mode == "think":
            m = re.search(r'<think>\n(.*?)\n</think>\n\n(.+)', content, re.DOTALL)
            if m:
                answer = m.group(2).strip()
                return answer if answer else None
            return None
        elif think_mode == "no_think":
            m = re.search(r'<think>\n\n</think>\n\n(.+)', content, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                return answer if answer else None
            return None
        else:
            return None

    def _parse_listwise_id_only(self, s: str):
        """
        解析格式如 "[1] > [2] = [3] > [4]"，返回列表，列表元素是列表（同级文档id按序号升序）
        例如： [[1], [2,3], [4]]
        """
        s = s.strip()
        if not s:
            return None
        groups = [g.strip() for g in s.split('>')]
        result = []
        for group in groups:
            # group内用=连接，且相同排名时序号小的在前面
            docs = [int(m) for m in re.findall(r'\[(\d+)\]', group)]
            if not docs:
                return None
            # 按序号升序排序
            # docs.sort()
            result.append(docs)
        return result

    def _parse_listwise_id_score(self, s: str):
        """
        解析格式如 "[1](3.0) > [2](2.0) = [3](2.0) > [4](1.0)"，
        返回两个字典：
        - order: [[1], [2,3], [4]]，同id_only格式
        - scores: {1:3.0, 2:2.0, 3:2.0, 4:1.0}
        """
        s = s.strip()
        if not s:
            return None, None
        groups = [g.strip() for g in s.split('>')]
        order = []
        scores = {}
        for group in groups:
            # group内用=连接
            parts = [p.strip() for p in group.split('=')]
            docs = []
            for part in parts:
                m = re.match(r'\[(\d+)\]\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)', part)
                if not m:
                    return None, None
                doc_id = int(m.group(1))
                score = float(m.group(2))
                docs.append(doc_id)
                scores[doc_id] = score
            # docs.sort()
            order.append(docs)
        return order, scores

    def _listwise_ndcg(self, pred_order, true_order, true_scores=None):
        # 展平
        pred_list = [doc for group in pred_order for doc in group]
        true_list = [doc for group in true_order for doc in group]

        max_rel = len(true_list)

        if true_scores is not None:
            true_scores_list = [true_scores[doc] for group in true_order for doc in group]
            if len(true_scores_list) != len(true_list):
                return 0.0
        else:
            true_scores_list = []
            max_rel = len(true_order)
            for i, group in enumerate(true_order):
                rel = max_rel - i
                for doc in group:
                    true_scores_list.append(rel)

        rel_dict = {
            doc: (max_rel - i) * float(score)
            for i, (doc, score) in enumerate(zip(true_list, true_scores_list))
        }

        def dcg(scores):
            return sum(
                (2 ** rel_dict.get(doc, 0) - 1) / math.log2(i + 2)
                for i, doc in enumerate(scores)
            )

        dcg_val = dcg(pred_list)
        idcg_val = dcg(true_list)

        return dcg_val / idcg_val if idcg_val > 0 else 0.

    def _listwise_score_mse(self, scores_pred, scores_true):
        """
        计算两个分数字典的均方误差，keys取交集
        """
        if set(scores_pred.keys()) != set(scores_true.keys()):
            return 0.0
        keys = set(scores_pred.keys()) & set(scores_true.keys())
        if not keys:
            return 1.0
        mse = sum(self._pointwise_score_mse(scores_pred[k], scores_true[k]) for k in keys) / len(keys)
        return mse


# 注册插件
orms['think_rank_reward'] = ThinkRankReward
orms['think_format_reward'] = ThinkFormatReward