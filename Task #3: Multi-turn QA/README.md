# Task 3: Multi-turn QA Pipeline

This directory contains the complete pipeline for Task 3 Multi-turn Question Answering, which involves data synthesis, retrieval, reranking, and model training.

## Overview

The Task 3 pipeline consists of four main stages:
1. **Data Retrieval** (`0_recall.py`) - Generate search queries and retrieve relevant information
2. **Reranking & Labeling** (`1_rerank.py`) - Rerank retrieved content and label relevance
3. **Data Synthesis** (`2_syth_data.py`) - Generate synthetic training data using LLM
4. **Model Training** (`src/3_train.sh`) - Fine-tune the model with LoRA

## Pipeline Stages

### Stage 1: Data Retrieval (`0_recall.py`)

**Purpose**: Generate search queries and retrieve relevant web content for each question-image pair.

**Key Features**:
- Loads original multi-turn QA data
- Generates search queries using GPT-4.1 based on query, image, and conversation history
- Performs web search using UnifiedSearchPipeline
- Downloads and processes images with caching
- Judges relevance of retrieved content using GPT-4o-mini

**Input**: `crag_mm_mt.jsonl` (original data)
**Output**: `crag_mm_mt_task3_webv6_recall30_labeled.jsonl` (data with retrieved content)

**Usage**:
```bash
python 0_recall.py
```

### Stage 2: Reranking & Labeling (`1_rerank.py`)

**Purpose**: Rerank retrieved content and label which information is relevant to the query.

**Key Features**:
- Uses TF-IDF and cosine similarity to find most relevant text chunks
- Asynchronously processes data for efficiency
- Judges relevance of each retrieved snippet using GPT-4o-mini
- Filters out irrelevant content
- Supports batch processing with error handling

**Input**: `crag_mm_mt_task3_webv6_recall30_labeled.jsonl`
**Output**: `crag_mm_mt_task3_webv6_recall30_pointwise_labeled.jsonl`

**Usage**:
```bash
python 1_rerank.py
```

### Stage 3: Data Synthesis (`2_syth_data.py`)

**Purpose**: Generate synthetic training data using Llama-3.2-11B-Vision-Instruct model.

**Key Features**:
- Loads VLLM model for efficient inference
- Generates multiple answer versions for each question
- Evaluates answer quality using GPT-4o-mini
- Filters and deduplicates data
- Splits data into train/validation sets

**Input**: `crag_mm_st_task1.jsonl` (original single-turn data)
**Output**: 
- `train_crag_mm_comb_task3_new_prompt_syth_data_fold0.jsonl`
- `eval_crag_mm_comb_task3_new_prompt_syth_data_fold0.jsonl`

**Usage**:
```bash
python 2_syth_data.py
```

### Stage 4: Model Training (`src/3_train.sh`)

**Purpose**: Fine-tune the Llama-3.2-11B-Vision-Instruct model using LoRA.

**Key Features**:
- LoRA fine-tuning with configurable parameters
- DeepSpeed integration for efficient training
- Multi-GPU training support
- Automatic checkpointing and evaluation
- Configurable hyperparameters

**Input**: Synthetic training data from Stage 3
**Output**: Fine-tuned model checkpoints

**Usage**:
```bash
cd src && bash 3_train.sh
```

## Configuration

### Model Paths
- **Base Model**: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Tokenizer**: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Checkpoint Path**: `/checkpoints`

### Data Paths
- **Training Data**: `/dataset/v0.1.2/comb`
- **Images**: `/dataset/v0.1.2/comb/images`

### Training Parameters
- **Learning Rate**: 5e-5
- **LoRA Rank**: 128
- **LoRA Alpha**: 256
- **LoRA Dropout**: 0.05
- **Epochs**: 10
- **Batch Size**: 1 (with gradient accumulation of 8)
- **Warmup Ratio**: 0.05

## Dependencies

### Python Packages
```bash
pip install torch vllm transformers openai tqdm scikit-learn numpy pillow requests
```

### Environment Setup
```bash
source activate YOUR_ENV
```

## File Structure

```
Task #3: Multi-turn QA/
├── README.md                           # This file
├── 0_recall.py                        # Stage 1: Data retrieval
├── 1_rerank.py                        # Stage 2: Reranking & labeling
├── 2_syth_data.py                     # Stage 3: Data synthesis
├── 3_train.py                         # Stage 4: Model training
├── src/
│   └── train_llama32_single_turn_lora.sh  # Original training script
└── 合成task3数据.ipynb                # Original Jupyter notebook
```

## Execution Order

1. **Prepare Data**: Ensure all input files are in the correct locations
2. **Run Retrieval**: `python 0_recall.py`
3. **Run Reranking**: `python 1_rerank.py`
4. **Run Synthesis**: `python 2_syth_data.py`
5. **Run Training**: `bash src/3_train.sh`

## Output Files

### Intermediate Files
- `crag_mm_mt_task3_webv6.jsonl` - First stage results
- `crag_mm_mt_task3_webv6_recall30_labeled.jsonl` - Retrieval results
- `crag_mm_mt_task3_webv6_recall30_pointwise_labeled.jsonl` - Reranking results

### Final Files
- `train_crag_mm_comb_task3_new_prompt_syth_data_fold0.jsonl` - Training data
- `eval_crag_mm_comb_task3_new_prompt_syth_data_fold0.jsonl` - Validation data
- Model checkpoints in the specified output directory

## Notes

- All scripts include comprehensive error handling and progress tracking
- The pipeline supports both single-turn and multi-turn QA scenarios
- GPU memory utilization is optimized for efficient training
- The synthesis stage generates multiple versions of each answer for better quality
- Evaluation is performed using GPT-4o-mini for consistency and accuracy

## Troubleshooting

1. **Memory Issues**: Reduce batch size or gradient accumulation steps
2. **API Rate Limits**: Add delays between API calls in retrieval and evaluation stages
3. **Model Loading**: Ensure sufficient GPU memory for VLLM model loading
4. **Data Format**: Verify input data format matches expected schema

For additional support, refer to the individual script documentation or contact the development team. 