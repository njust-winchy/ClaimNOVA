# ClaimNOVA: Evidence-grounded Claim-level Novelty Evaluation for Academic Papers
<img width="2968" height="876" alt="overview" src="https://github.com/user-attachments/assets/931a1525-e2ba-4d1b-8ccd-276ef078a906" />


This is the official repository for the dataset and code of the paper: "ClaimNOVA: Evidence-grounded Claim-level Novelty Evaluation for Academic Papers".<br>

## Overview
- We propose ClaimNOVA, a type-aware and evidence-grounded framework for claim-level novelty judgment. It decomposes novelty assessment into evidence retrieval, claim-level judgment, and structured evaluation. <br>
- We use novelty type as a structured prior for evidence selection and type-aware comparison. This helps compare different claims with more appropriate prior work. <br>
- Experiments on NovBench show that \textsc{ClaimNOVA} improves alignment with source novelty claims and coverage of human reviewer evaluations over zero-shot, few-shot, and RAG baselines. Further analyses reveal the roles of evidence grounding, novelty type, and claim-level evaluation, as well as trade-offs introduced by retrieval-augmented evidence..
## Dataset
The dataset can be obtained from here ([https://drive.google.com/drive/folders/1VZTCcUngoBa4jC3skKgbcefyotuHY9Zc?usp=drive_link](https://drive.google.com/file/d/17eCOU_WUUYIlg7V_hf1ERcDX7pZxiBs_/view?usp=drive_link)).<br>
Data on ACL Anthology can be obtained here: https://github.com/tangg555/acl-anthology-helper.<br>
<pre>
ClaimNOVA                                     Root directory
├── Evaluation.py                             Code for processing the result
├── GPT4type.py                               Code for processing the novelty type
├── build_background.py                       Code for building backgrround
├── build_background_card.py                  Code for building backgrround card
├── export_candidated.py                      Code for export candidated paper
├── run_ablation_experiments.py               Code for ablation study
├── run_background_evidence_experiments.py    Code for background evidence experiments
├── run_multi_agent_novelty_pipeline.py       Code for multi agent novelty pipeline
├── run_novelty_type_stratified_analysis.py   Code for novelty type stratified analysis
└── README.md

</pre>
## Run
python build_background_llm.py

python build_background_card.py \
  --input_dir /background_raw_store \
  --output_dir /background_card_store \
  --card_model /local_model/Qwen3-14B \
  --embed_model /all-MiniLM-L6-v2 \
  --tensor_parallel_size 1 \
  --max_model_len 8192 \
  --max_new_tokens 256 \
  --llm_batch_size 8 \
  --trust_remote_code

python run_multi_agent_novelty_pipeline_v7.py \
  --target_papers_json /dataset.json \
  --background_card_dir /background_card_store \
  --output_dir /multi_agent_outputs \
  --model /local_model/Qwen3-14B \
  --tensor_parallel_size 1 \
  --max_model_len 8192 \
  --max_new_tokens 1024 \
  --selector_prefilter_k 12 \
  --selector_final_k 5 \
  --trust_remote_code

## Dependency packages
System environment is set up according to the following configuration:
- transformers==4.56.2
- nltk==3.6.7
- matplotlib==3.5.1
- scikit-learn==1.1.3
- pytorch==2.10.0
- tqdm==4.65.0
- numpy==1.24.1
- pandas==2.2.3
- openai==1.53.0
- vllm==0.19.0
- sentence-transformers==5.3.0
- ai_researcher
- PyMuPDF=1.27.2.2
- scientific-information-change
- langchain=1.2.14
## Acknowledgement
ClaimNOVA is intended to assist reviewers, not replace them. 
## Citation
Please cite the following paper if you use this code and dataset in your work.
