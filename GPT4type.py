# import json
#
# import re
#
# def parse_llm_json(text):
#     """
#     解析 LLM 返回的 ```json {...} ``` 结构
#     """
#
#     # 提取 code block
#     match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
#
#     if match:
#         json_text = match.group(1)
#     else:
#         # 如果没有 ```json，就找 {...}
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if not match:
#             return None
#         json_text = match.group()
#
#     try:
#         data = json.loads(json_text)
#         return data
#     except json.JSONDecodeError:
#         return None
#
#
#
# with open('Novelty_types.json', encoding='utf-8') as f:
#     json_data = json.load(f)
# f.close()
#
# novelty_type = []
# final_list = []
# for i in json_data:
#     if len(i['paper_novelty'])==0 or len(i['review_novelty'])==0:
#         continue
#     result = parse_llm_json(i['output'])
#     for j in result['final_novelty_type']:
#         novelty_type.append(j)
#     i['novelty_type_preliminary'] = result
#     i.pop('output')
#     final_list.append(i)
#
# with open('Novelty_types_new.json', 'w', encoding='utf-8') as f:
#     json.dump(final_list, f, ensure_ascii=False, indent=4)
#     f.close()
# print()

# from openai import OpenAI
# from tqdm import tqdm
# import json
# import tiktoken
# encoding = tiktoken.encoding_for_model("gpt-4o")
#
# def build_prompt(intro_novelty_text, review_novelty_text):
#     prompt = f"""Identify the broad novelty type(s) of the given paper based on:
# (1) novelty-related descriptions extracted from the introduction, and
# (2) reviewer comments about the paper's novelty.
#
# The two sources refer to the same paper.
#
# Instructions:
# - Infer the novelty type(s) from each source separately, then give the final novelty type(s).
# - Use broad, high-level, and reusable type names only.
# - Do not use overly specific or paper-specific descriptions.
# - Each type name should be short, ideally 1–3 words.
# - Do not include sentiment or quality judgments.
# - Return the output as a json dictionary.
#
# Output format:
# {{
#   "intro_novelty_type": ["..."],
#   "review_novelty_type": ["..."],
#   "final_novelty_type": ["..."]
# }}
#
# Introduction novelty descriptions:
# {intro_novelty_text}
#
# Reviewer novelty comments:
# {review_novelty_text}
# """
#     return prompt
#
# def gpt4task(prompt, model):
#     client = OpenAI(
#
#     )
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 'role': 'user',
#                 'content': prompt,
#
#             }
#         ],
#         model=model,
#         temperature=0,
#     )
#     return chat_completion.choices[0].message.content
#
# with open('EMNLP_23_24.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
# f.close()
# save_list = []
# for item in tqdm(data):
#     intro_novelty_text = item["paper_novelty"]
#     review_novelty_text = item["review_novelty"]
#     input_prompt = build_prompt(intro_novelty_text, review_novelty_text)
#     output = gpt4task(input_prompt, 'gpt-4o')
#     item['output'] = output
#     save_list.append(item)
#
# with open('Novelty_types.json', 'w', encoding='utf-8') as f:
#     json.dump(save_list, f, ensure_ascii=False, indent=4)
#     f.close()

from openai import OpenAI
from tqdm import tqdm
import json
import os
import time

SAVE_PATH = "Gpt4types.json"
CHECKPOINT = 100


def build_prompt(intro_novelty_text):
    prompt = f"""You are an expert NLP researcher.

Your task is to identify the MAIN types of novelty in a paper.

IMPORTANT REQUIREMENTS:

You MUST choose types ONLY from the following list:
["method", "dataset", "task", "application", "theory", "result"]
Each output type must be:
High-level (NOT too specific)
Conceptually distinct
Output 1–3 types ONLY.
If multiple types apply, include all major ones, but do NOT exceed 3.
Do NOT generate new labels or synonyms.
For example, use "method" instead of "model", "framework", or "approach"
Use "dataset" instead of "data" or "corpus"
If unsure, prefer:
"method" for modeling or algorithmic contributions
"dataset" for data construction or resources
"task" for new problem formulations
"application" for real-world or domain-specific usage
"theory" for analytical or explanatory contributions
"result" for performance improvements or empirical findings
Do NOT provide any explanation.

OUTPUT FORMAT (strict JSON):
{{
"novelty_types": ["type1", "type2"]
}}
    
    Introduction novelty descriptions:
    {intro_novelty_text}
    
    """
    return prompt


def gpt4task(prompt, model):
    client = OpenAI(
        base_url='',
        # required but ignored
        api_key='',
        timeout=120,
        max_retries=3
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,

            }
        ],
        model=model,
        temperature=0,

    )
    return chat_completion.choices[0].message.content


# 读取数据
with open('Novelty_types_new.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 如果已有结果文件 → 断点续跑
save_list = []
start_idx = 0

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        save_list = json.load(f)
        start_idx = len(save_list)
        print(f"Resuming from {start_idx}")

# 主循环
for i in tqdm(range(start_idx, len(data))):

    item = data[i]

    intro_novelty_text = item["paper_novelty"]
    #review_novelty_text = item["review_novelty"]

    prompt = build_prompt(intro_novelty_text)


    output = gpt4task(prompt, 'gpt-5.4')
    time.sleep(0.2)

    item["output"] = output
    save_list.append(item)

    # 每100条保存一次
    if len(save_list) % CHECKPOINT == 0:
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(save_list, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(save_list)} items")

# 最终保存
with open(SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(save_list, f, ensure_ascii=False, indent=4)

print("Finished!")