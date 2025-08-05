import os
import json
from tqdm.auto import tqdm
from pathlib import Path
import pytrec_eval
from decimal import Decimal, ROUND_HALF_UP

def round_half_up(n, decimals=1):
    d = Decimal(str(n))
    return float(d.quantize(Decimal('1.' + '0'*decimals), rounding=ROUND_HALF_UP))

def list_jsonl_files(dir_path):
    p = Path(dir_path)
    return [str(f.resolve()) for f in p.glob("*.jsonl")]

def load_anserini_qrels(split, base_dir):
    # split: 'dl19-passage' 或 'dl20-passage'
    qrels_file = os.path.join(base_dir, 'topics-and-qrels', f'qrels.{split}.txt')
    # topics_file = os.path.join(base_dir, 'topics-and-qrels', f'topics.{split}.txt')
    
    # # 加载 topics
    topics = {}
    # with open(topics_file, encoding='utf8') as f:
    #     for line in f:
    #         qid, qtext = line.strip().split(None, 1)
    #         topics[qid] = qtext
    
    # 加载 qrels
    qrels = {}
    with open(qrels_file, encoding='utf8') as f:
        for qid, _, docid, rel in (l.strip().split() for l in f):
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    
    return topics, qrels

def load_bright_qrels(split, qrels_dir):
    qrel = {}
    qrel_file = os.path.join(qrels_dir, f'{split}.tsv')
    if os.path.exists(qrel_file):
        with open(qrel_file, 'r', encoding='utf-8') as f_qr:
            for line in f_qr:
                parts = line.strip().split()
                if len(parts) < 4: continue
                qid, _, docid, rel = parts[:4]
                qrel.setdefault(qid, {})[docid] = int(rel)
    return {
        "qrels": qrel,
    }

def load_beir_with_anserini(split, qrels_dir):
    topics, qrels = load_anserini_qrels(f'beir-v1.0.0-{split}.test', qrels_dir)
    return {
        "queries": topics,
        "qrels": qrels,
        "documents": None,
        "excluded_ids": {},
    }

def compute_metrics(
    qrels: dict,
    results: dict,
    k_values: tuple = (5, 10, 50, 100, 200, 1000)
) -> dict:
    ndcg, _map, recall = {}, {}, {}
    for k in k_values:
        _map[f"MAP@{k}"] = 0.0
        ndcg[f"NDCG@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)
    for query_id in scores:
        for k in k_values:
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 4) for k, v in m.items()}
    _map = _normalize(_map)
    ndcg = _normalize(ndcg)
    recall = _normalize(recall)
    all_metrics = {}
    for mt in [_map, ndcg, recall]:
        all_metrics.update(mt)
    return all_metrics

def print_table(all_results, title, f, order, name_map, round_digits=1):
    print('*******************')
    print(f'{title}!!!!!')
    f.write('*******************\n')
    f.write(f'{title}!!!!!\n')
    values = []
    valid_vals = []
    for display_name in order:
        task_name = name_map[display_name]
        val = all_results.get(task_name, None)
        if val is not None:
            percent_val = val * 100
            rounded_val = round_half_up(percent_val, round_digits)
            values.append(f"{rounded_val:.{round_digits}f}")
            valid_vals.append(rounded_val)
        else:
            values.append("N/A")
    if valid_vals:
        avg_val = round_half_up(sum(valid_vals) / len(valid_vals), round_digits)
        values.append(f"{avg_val:.{round_digits}f}")
    else:
        values.append("N/A")
    line = " & ".join(values) + " \\\\"
    col_widths = [max(len(o), len(v)) for o, v in zip(order, values)]
    fmt = " ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*order))
    print(fmt.format(*values))
    print('copy')
    f.write(fmt.format(*order) + '\n')
    f.write(fmt.format(*values) + '\n')
    return line

def get_result_filename(result_dir):
    # 去掉末尾斜杠
    result_dir = os.path.normpath(result_dir)
    # 拆分路径
    parts = result_dir.split(os.sep)
    # 取最后3级
    last3 = parts[-3:]
    # 拼接
    filename = '.'.join(last3) + '.eval_results.txt'
    return filename

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True, help='预测结果jsonl目录')
    parser.add_argument('--qrels_dir', type=str, required=True, help='qrels目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--config', type=str, required=True, help='order和name_map的json配置文件')
    parser.add_argument('--round_digits', type=int, default=1, help='四舍五入位数')
    parser.add_argument('--count_line', type=str2bool, default=False)
    args = parser.parse_args()

    # 读取config
    with open(args.config, 'r', encoding='utf-8') as cf:
        config = json.load(cf)
    order = config['order']
    name_map = config['name_map']

    jsonl_list = list_jsonl_files(args.result_dir)
    jsonl_list.sort()
    qrels_dir = args.qrels_dir

    all_label_results = {}
    all_yesno_results = {}
    all_results = {}

    for input_file_name in tqdm(jsonl_list):
        print(input_file_name)
        base_name = os.path.basename(input_file_name)
        dataset_name = os.path.splitext(base_name)[0]
        task = dataset_name.split('.')[0]
        if 'BRIGHT' in input_file_name:
            qrels = load_bright_qrels(task, qrels_dir=qrels_dir)["qrels"]
        elif 'BEIR' in input_file_name:
            qrels = load_beir_with_anserini(task, qrels_dir=qrels_dir)["qrels"]

        with open(input_file_name, 'r', encoding='utf-8') as f:
            L = f.readlines()
        if args.count_line:
            print(input_file_name,len(L))
        data = [json.loads(i) for i in L]

        label_results = {}
        yesno_results = {}
        results = {}
        for item in data:
            qid = str(item['qid'])
            docid = str(item['docid'])
            try:
                label_score = float(item['predicted_label_avg'])
            except:
                label_score = None
            yesno_score = float(item['predicted_yesno_score'])
            score_list = [label_score / 4.0, yesno_score] if label_score is not None else [yesno_score]
            clean = [x for x in score_list if x is not None]
            score = sum(clean) / len(clean)
            if qid not in label_results:
                label_results[qid] = {}
            if qid not in yesno_results:
                yesno_results[qid] = {}
            if qid not in results:
                results[qid] = {}
            label_results[qid][docid] = label_score
            yesno_results[qid][docid] = yesno_score
            results[qid][docid] = score

        try:
            label_metrics = compute_metrics(qrels, label_results)
            ndcg_10 = label_metrics.get('NDCG@10', None)
            if ndcg_10 is not None:
                all_label_results[task] = ndcg_10
        except Exception as e:
            print(f"Label metric error for {task}: {e}")

        try:
            yesno_metrics = compute_metrics(qrels, yesno_results)
            ndcg_10 = yesno_metrics.get('NDCG@10', None)
            if ndcg_10 is not None:
                all_yesno_results[task] = ndcg_10
        except Exception as e:
            print(f"YesNo metric error for {task}: {e}")

        try:
            merge_metrics = compute_metrics(qrels, results)
            ndcg_10 = merge_metrics.get('NDCG@10', None)
            if ndcg_10 is not None:
                all_results[task] = ndcg_10
        except Exception as e:
            print(f"Merge metric error for {task}: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, get_result_filename(args.result_dir))

    with open(output_file, "w", encoding="utf-8") as f:
        line1 = print_table(all_label_results, "Label", f, order, name_map, args.round_digits)
        line2 = print_table(all_yesno_results, "YesNo", f, order, name_map, args.round_digits)
        line3 = print_table(all_results, "Merge", f, order, name_map, args.round_digits)
        print(line1)
        print(line2)
        print(line3)
        f.write(line1 + '\n')
        f.write(line2 + '\n')
        f.write(line3 + '\n')

    print(f"\n结果已写入: {output_file}")

if __name__ == "__main__":
    main()