# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch
import os
import pandas as pd
import numpy as np
import ast
from collections import Counter
import math
import re
from scientific_information_change.estimate_similarity import SimilarityEstimator
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== Load language model once (outside loop) ======
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
w_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to('cuda')
w_model.eval()
estimator = SimilarityEstimator()

def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to('cuda')
    with torch.no_grad():
        outputs = w_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()



def evaluate_distribution_and_kl(y_true, y_pred, labels=None):
    """

      - distribution_accuracy

    """

    # 自动获取标签集合
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    true_cnt = Counter(y_true)
    pred_cnt = Counter(y_pred)

    total_true = sum(true_cnt.values())
    total_pred = sum(pred_cnt.values())

    # 1. 类别分布
    distribution_true = {lab: true_cnt[lab] / total_true for lab in labels}
    distribution_pred = {lab: pred_cnt[lab] / total_pred for lab in labels}

    # 2. 分布相似度 (1 - L1_norm / 2)
    l1 = sum(abs(distribution_true[lab] - distribution_pred[lab]) for lab in labels)
    distribution_accuracy = 1 - l1 / 2

    return {
        "distribution_accuracy": distribution_accuracy,
    }
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
def evaluate_single_set(intro_sentences, evals, gold_evals=None, model=None):
    """
    ��ĳһ������ (ר�� �� ģ��) ���Զ�����
    intro_sentences: ������ӱ�Ծ���
    evals: list of (label, text)  �� list of str (û�б�ǩʱ)
    gold_evals: gold ��ע (ר������), ���� correctness �Ա�
    """
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # ���������Ƿ����ǩ
    if isinstance(evals[0], tuple):
        labels, texts = zip(*evals)
    else:
        labels, texts = [None] * len(evals), evals

    # # ---- 1. Relevance ----
    intro_emb = model.encode(" ".join(intro_sentences), convert_to_numpy=True)
    eval_embs = model.encode(texts, convert_to_numpy=True)
    relevance = estimator.estimate_ims(intro_sentences, texts)
    relevance = relevance.max(axis=1)
    relevance = relevance.mean()
    # ---- 2. Correctness ----
    if gold_evals is None:  #
        gold_labels = [lbl for lbl, _ in evals]
        correctness = evaluate_distribution_and_kl(gold_labels, gold_labels)
    else:
        gold_labels = [lbl for lbl, _ in gold_evals]
        pred_labels = [t for t in labels]
        correctness = evaluate_distribution_and_kl(gold_labels, pred_labels)


    # # ---- 3. Coverage ----
    if gold_evals is None:  # ר���Լ����Լ� = 100%
        coverage = 1.0
    else:
        gold_texts = [txt for _, txt in gold_evals]
        gold_embs = model.encode(gold_texts, convert_to_numpy=True)
        coverage_hits = 0
        for g in gold_embs:
            sims = [cosine_similarity(g, m) for m in eval_embs]
            if max(sims) >= 0.7:
                coverage_hits += 1
        coverage = coverage_hits / len(gold_embs) if len(gold_embs) > 0 else 0.0

    # ---- 4. Clarity ----
    # texts: generated evaluation sentences
    # intro_sentences: introduction sentences

    # ---- Keyword Coverage (KC) ----
    keywords = [
        w for s in intro_sentences
        for w in re.findall(r"\b[A-Za-z0-9\-]+\b", s)
        if len(w) > 5
    ]

    keyword_count = sum(
        any(k.lower() in t.lower() for k in keywords)
        for t in texts
    )
    keyword_ratio = keyword_count / len(texts) if texts else 0.0

    # ---- Length Score (LS) ----
    avg_len = np.mean([len(t.split()) for t in texts]) if texts else 0
    length_score = min(avg_len / 20, 1)

    # ---- Fluency Score (FS) ----
    if texts:
        perplexities = [compute_perplexity(t) for t in texts]
        avg_ppl = np.mean(perplexities)

        # Normalize inverse perplexity
        fluency_score = 1 / (1 + avg_ppl)
    else:
        fluency_score = 0.0

    # ---- Final Clarity ----
    clarity = (keyword_ratio + length_score + fluency_score) / 3

    return {
        "Relevance": float(relevance),
        "Correctness": correctness,
        "Coverage": coverage,
        "Clarity": clarity
    }

model = SentenceTransformer("all-MiniLM-L6-v2")

# takes a list of sentences A of length N and a list of sentences B of length M and returns a numpy array S of size N×M,
# where S_{ij} is the IMS between A_i and B_j.
def load_dict(x):
    try:
        return ast.literal_eval(x)
    except:
        return None  # 解析失败时返回 None
def transfer(data):
    expert = []

    for label, items in data.items():
        for content in items:
            if not isinstance(content, str):
                continue
            c = content.strip()
            # 过滤 None、"None" 和长度 < 10 的内容
            if c.lower() == "none" or len(c) < 10:
                continue
            expert.append((label, c))
    return expert

def extract_novelty_evaluations(text):
    """
    从文本中抽取 Positive / Neutral / Negative Novelty Evaluations 段落内容
    返回 dict: {"Positive": [...], "Neutral": [...], "Negative": [...]}
    若某类不存在，则值为 None
    """
    pattern = re.compile(
        r"(Positive|Neutral|Negative)\s+Novelty\s+Evaluations\s*(.*?)(?=(?:Positive|Neutral|Negative)\s+Novelty\s+Evaluations|$)",
        re.S
    )
    matches = pattern.findall(text)

    results = defaultdict(list)
    for label, content in matches:
        # 提取以“-”开头的条目
        items = re.findall(r"-\s*(.+)", content)
        results[label].extend(item.strip() for item in items if item.strip())

    # 确保三个类别都存在；无内容则为 None
    final_results = {}
    for label in ["Positive", "Neutral", "Negative"]:
        if label in results and results[label]:
            final_results[label] = results[label]
        else:
            final_results[label] = ['None']

    return final_results

with open('Dataset_with_type_v2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    f.close()

# #Novelty type
# file_list = os.listdir('novelty_type_analysis/per_type')
# for file_name in file_list:
#     result = []
#     with open('novelty_type_analysis/per_type/'+file_name+'/predictions_subset.json', 'r', encoding='utf-8') as f:
#         type_results = json.load(f)
#     f.close()
#     save_name = file_name + '.json'
#     for i in type_results:
#         for j in data:
#             if j['paper_id'] == i['paper_id']:
#                 intro = j['paper_novelty']
#                 expert_evals = extract_novelty_evaluations(j['output_format'])
#         model_evals = extract_novelty_evaluations(i['final_review_text'])
#         model_output = transfer(model_evals)
#         expert = transfer(expert_evals)
#         model_result = evaluate_single_set(intro, model_output, gold_evals=expert, model=model)
#         result.append({'Model': model_result})
#
#     with open(save_name, 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=4, ensure_ascii=False)
#     f.close()

#ablation process
file_list = os.listdir('ablation_outputs')
for file in file_list:
    result = []
    if file.endswith('.json'):
        continue
    save_name = 'ablation_'+file+'.json'
    if os.path.exists(save_name):
        continue
    for res in tqdm(data):
        if len(res) < 12:
            continue
        sample_id = res['paper_id']
        intro = res['paper_novelty']
        expert_evals = extract_novelty_evaluations(res['output_format'])

        llm_output_path = 'ablation_outputs/'+file+'/' + str(sample_id) + '/' + 'writer.json'
        if not os.path.exists(llm_output_path):
            continue
        with open(llm_output_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
            f.close()
        model_evals = {}
        model_evals['Positive'] = llm_data['positive_novelty_evaluations']
        model_evals['Neutral'] = llm_data['neutral_novelty_evaluations']
        model_evals['Negative'] = llm_data['negative_novelty_evaluations']
        # model_evals = extract_novelty_evaluations(llm_data['final_review_text'])
        model_output = transfer(model_evals)
        expert = transfer(expert_evals)

        model_result = evaluate_single_set(intro, model_output, gold_evals=expert, model=model)
        result.append({'Model': model_result})

    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()
    print()


#background process
file_list = os.listdir('background_evidence_outputs')
for file in file_list:
    result = []
    if file.endswith('.json'):
        continue
    save_name = 'background_'+file+'.json'
    if os.path.exists(save_name):
        continue
    for res in tqdm(data):
        if len(res) < 12:
            continue
        sample_id = res['paper_id']
        intro = res['paper_novelty']
        expert_evals = extract_novelty_evaluations(res['output_format'])

        llm_output_path = 'background_evidence_outputs/'+file+'/' + str(sample_id) + '/' + 'writer.json'
        if not os.path.exists(llm_output_path):
            continue
        with open(llm_output_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
            f.close()
        model_evals = {}
        model_evals['Positive'] = llm_data['positive_novelty_evaluations']
        model_evals['Neutral'] = llm_data['neutral_novelty_evaluations']
        model_evals['Negative'] = llm_data['negative_novelty_evaluations']
        # model_evals = extract_novelty_evaluations(llm_data['final_review_text'])
        model_output = transfer(model_evals)
        expert = transfer(expert_evals)

        model_result = evaluate_single_set(intro, model_output, gold_evals=expert, model=model)
        result.append({'Model': model_result})

    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()
    print()

# #single file process
# save_name = 'ablation_full.json'
# for res in tqdm(data):
#     if len(res)<12:
#         continue
#     sample_id = res['paper_id']
#     intro = res['paper_novelty']
#     expert_evals = extract_novelty_evaluations(res['output_format'])
#
#     llm_output_path = 'ablation_outputs/full/'+str(sample_id)+'/'+'writer.json'
#     if not os.path.exists(llm_output_path):
#         continue
#     with open(llm_output_path, 'r', encoding='utf-8') as f:
#         llm_data = json.load(f)
#         f.close()
#     model_evals = {}
#     model_evals['Positive'] = llm_data['positive_novelty_evaluations']
#     model_evals['Neutral'] = llm_data['neutral_novelty_evaluations']
#     model_evals['Negative'] = llm_data['negative_novelty_evaluations']
#     #model_evals = extract_novelty_evaluations(llm_data['final_review_text'])
#     model_output = transfer(model_evals)
#     expert = transfer(expert_evals)
#
#     model_result = evaluate_single_set(intro, model_output, gold_evals=expert, model=model)
#     result.append({'Model': model_result})
#
# with open(save_name, 'w', encoding='utf-8') as f:
#     json.dump(result, f, indent=4, ensure_ascii=False)
# f.close()
# print()
