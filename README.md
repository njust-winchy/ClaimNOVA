# ClaimNOVA: Evidence-grounded Claim-level Novelty Evaluation for Academic Papers
<img width="2968" height="876" alt="overview" src="https://github.com/user-attachments/assets/931a1525-e2ba-4d1b-8ccd-276ef078a906" />


This is the official repository for the dataset and code of the paper: "ClaimNOVA: Evidence-grounded Claim-level Novelty Evaluation for Academic Papers".<br>

## Overview
- We propose ClaimNOVA, a type-aware and evidence-grounded framework for claim-level novelty judgment. It decomposes novelty assessment into evidence retrieval, claim-level judgment, and structured evaluation. <br>
- We use novelty type as a structured prior for evidence selection and type-aware comparison. This helps compare different claims with more appropriate prior work. <br>
- Experiments on NovBench show that \textsc{ClaimNOVA} improves alignment with source novelty claims and coverage of human reviewer evaluations over zero-shot, few-shot, and RAG baselines. Further analyses reveal the roles of evidence grounding, novelty type, and claim-level evaluation, as well as trade-offs introduced by retrieval-augmented evidence..
## Dataset
The dataset can be obtained from here ([https://drive.google.com/drive/folders/1VZTCcUngoBa4jC3skKgbcefyotuHY9Zc?usp=drive_link](https://drive.google.com/file/d/17eCOU_WUUYIlg7V_hf1ERcDX7pZxiBs_/view?usp=drive_link)).<br>
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
│
└── README.md

</pre>
