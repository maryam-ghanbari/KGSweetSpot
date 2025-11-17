# KGSweetSpot

# Evaluating Knowledge Graph-Enhanced Context for Multiple-Choice QA

This repository accompanies the paper  
**“Evaluating Knowledge Graph-Enhanced Context for Multiple-Choice Question Answering” (ICKG 2025)**  
It studies how different aspects of knowledge integration—**relevance, extraction scope, serialization format, and processing strategy**—affect the reasoning performance of large language models in multiple-choice QA tasks.

---

## 🔧 Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download raw data (ConceptNet + CSQA)
bash ./download_raw_data.sh

# 3. Preprocess data
python ./preprocess.py --run common
python ./preprocess.py --run csqa

# 4. Extract and serialize subgraphs
python ./utils/get_knowledge.py \
  --output ./knowledge/concept_net.dev.csqa.json \
  --serialization triple \
  --scope both

# 5. Run inference with T5, and non-T5 models (Llama, Mistral,etc.)
CUDA_VISIBLE_DEVICES=0 python ./utils/infer_t5.py \
  --task csqa \
  --model-type google/flan-t5-small \
  --input-path ./knowledge/concept_net.dev.csqa.json

Overview

preprocess.py prepares data and grounds question/answer concepts to ConceptNet.

get_knowledge.py extracts subgraphs and converts them into triple or path-based text.

infer_t5.py runs inference using a T5 model and non-T5 models like Llama and Mistral and evaluates reasoning accuracy.

This setup allows analyzing how different knowledge integration factors influence model reasoning.

Parts of the preprocessing and subgraph extraction pipeline are adapted from
QA-GNN (Yasunaga et al., NAACL 2021)
.
