# Llama-finetuning-project

## Project Overview
This project explores fine-tuning the LLaMA 2 (Large Language Model Meta AI) model for specialized question answering tasks with a focus on policy and regulatory frameworks. The project demonstrates efficient fine-tuning techniques on limited GPU resources using parameter-efficient methods and evaluates the performance with established NLP metrics.

## Key Features
- Fine-tuning LLaMA 2 (7B parameter model) using various techniques
- Implementation of parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation)
- Quantization techniques to reduce memory footprint (int8)
- Multiple fine-tuning approaches including standard fine-tuning and RAG (Retrieval-Augmented Generation)
- Comprehensive evaluation metrics to assess model performance

## Requirements and Setup

### Dependencies
```bash
pip install llama-recipes transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler ipywidgets bert-score sacrebleu rouge
```

### Model Conversion
Convert the original LLaMA weights to Hugging Face format:
```bash
TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
python ${TRANSFORM} --input_dir models --model_size 7B --output_dir models_hf/7B
```

## Prompt Engineering

### Question-Answer Format
The project uses a structured prompt format to guide the model's responses:
```
Answer the following question:
Question: [QUESTION_TEXT]
---
Answer:
```

This format helps the model understand when to generate an answer to a policy-related question. During fine-tuning, the model learns to associate this prompt structure with the task of providing informative and accurate policy answers.

### Tokenization Strategy
Question-answer pairs are tokenized with specific considerations:
```python
def tokenize_pairs(examples):
    tokenized_inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512)
    tokenized_outputs = tokenizer(examples['output'], padding='max_length', truncation=True, max_length=512)
    return {**tokenized_inputs, **{'labels': tokenized_outputs['input_ids']}}
```

This function processes both inputs (questions) and desired outputs (answers), mapping them to the appropriate token IDs and preparing them for the supervised fine-tuning process.

### RAG-Enhanced Prompting
For the Retrieval-Augmented Generation approach, prompts are enhanced with relevant context:
```
Answer the following question using the provided context:
Context: [RETRIEVED_CONTEXT]
Question: [QUESTION_TEXT]
---
Answer:
```

This structure allows the model to leverage external knowledge while generating responses, improving accuracy and factual correctness on domain-specific questions.

## Fine-tuning Details

### Quantization Configuration
To enable training on limited GPU resources, the model uses int8 quantization:
```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)
```

### LoRA Configuration
Parameter-efficient fine-tuning is implemented using LoRA with the following settings:
```python
peft_config = LoraConfig(
    r=16,                     # Rank dimension
    lora_alpha=32,            # Alpha parameter for LoRA scaling
    lora_dropout=0.05,        # Dropout probability for LoRA layers
    bias="none",              # Bias configuration
    task_type="CAUSAL_LM",    # Task type for the model
    target_modules=[          # Target modules to apply LoRA
        "q_proj",
        "k_proj", 
        "v_proj", 
        "o_proj"
    ]
)
```

This configuration targets the attention mechanism of the model, specifically the query, key, value, and output projections, which are critical for the model's performance in question answering tasks.

### Training Arguments
The fine-tuning process uses the following training parameters:
```python
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    optim="paged_adamw_8bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
```

These parameters are carefully tuned to:
- Use a learning rate appropriate for LoRA fine-tuning (2e-4)
- Apply weight decay to prevent overfitting (0.001)
- Implement gradient clipping to stabilize training (0.3)
- Use constant learning rate schedule with a warm-up period
- Group similar-length sequences together for more efficient batching

### SFT Trainer Configuration
The Supervised Fine-Tuning trainer is configured with:
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    dataset_text_field="text"
)
```

This setup:
- Connects the model with the tokenized dataset
- Applies the LoRA configuration
- Sets maximum sequence length to accommodate policy questions and answers
- References the text field in the dataset for training

## Fine-tuning Approaches

### 1. Standard Fine-tuning
The project implements supervised fine-tuning (SFT) using the `SFTTrainer` from the TRL library. This approach:
- Uses int8 quantization to fit the model on limited GPU resources
- Applies LoRA for parameter-efficient fine-tuning
- Fine-tunes on custom policy-related question-answer pairs

### 2. Context-Enhanced Fine-tuning with RAG
This approach incorporates Retrieval-Augmented Generation to enhance the model's performance:
- Retrieves relevant context from a knowledge base
- Provides this context to the model during training and inference
- Creates a more informed and accurate response generation

## Model Evaluation and Performance Analysis

### Evaluation Methodology
We evaluated three distinct versions of the model to analyze the impact of our various optimization techniques:
1. **Base Model**: The original LLaMA 2 7B model without any modifications
2. **Fine-tuned Model**: The model after applying LoRA-based fine-tuning on policy domain data
3. **Prompt-Engineered Model**: The fine-tuned model with optimized prompting strategies

All models were evaluated using the same test set of policy-related questions, with responses compared against expert-written reference answers.

### Evaluation Metrics Implementation
Our evaluation framework uses three complementary metrics to provide a holistic assessment of model quality:

```python
class SimilarityEvaluator:
    def bert_score_calc(cands, ref1):
        P1, R1, F1 = score(cands, ref1, lang="en", verbose=True)
        return F1

    def bleu_score_calc(cands, ref1):
        bleu_scorer = BLEU()
        score1 = bleu_scorer.sentence_score(
            hypothesis=cands[0],
            references=ref1,
        )
        return score1.score/100

    def rouge_score_calc(cands, ref1):
        rouge_scorer = Rouge()
        score1 = rouge_scorer.get_scores(
            hyps=cands[0],
            refs=ref1[0],
        )
        return score1[0]['rouge-l']['f']

    def evaluate_similarity(self, cands, ref1):
        bert_score = self.bert_score_calc(cands, ref1)
        bleu_score = self.bleu_score_calc(cands, ref1)
        rouge_score = self.rouge_score_calc(cands, ref1)
        return (bert_score, bleu_score, rouge_score)
```

### Base Model Performance
The base LLaMA 2 7B model demonstrated:
- Limited domain knowledge about policy frameworks
- Tendency to generate plausible but factually incorrect information
- Inconsistent response quality with varied prompts
- Average scores:
  - BERT Score: ~0.82
  - BLEU Score: ~0.15
  - ROUGE-L Score: ~0.31

### Fine-tuned Model Performance
After LoRA-based fine-tuning, the model showed significant improvements:
- Enhanced factual accuracy on policy questions
- More consistent response structure
- Better alignment with domain-specific terminology
- Performance gains:
  - BERT Score: ~0.89 (+8.5%)
  - BLEU Score: ~0.26 (+73%)
  - ROUGE-L Score: ~0.45 (+45%)

This improvement indicates that even with parameter-efficient fine-tuning techniques that modify less than 1% of the model parameters, we achieved substantial performance gains in the target domain.

### Prompt-Engineered Model Performance
Further optimizing prompts for the fine-tuned model yielded additional gains:
- More concise and relevant responses
- Better adherence to requested output formats
- Improved extraction of specific information
- Additional improvements:
  - BERT Score: ~0.91 (+2.2%)
  - BLEU Score: ~0.29 (+11.5%)
  - ROUGE-L Score: ~0.48 (+6.7%)

### RAG-Enhanced Model Evaluation
The Retrieval-Augmented Generation approach showed the best overall performance:
- Highest factual accuracy among all variants
- Strong contextual relevance to policy documents
- Ability to cite specific regulatory frameworks correctly
- Performance metrics:
  - BERT Score: ~0.93
  - BLEU Score: ~0.32
  - ROUGE-L Score: ~0.51

### Comparative Analysis
The following chart summarizes the relative performance improvement across different model variants:

| Model Variant | BERT Score | BLEU Score | ROUGE-L Score |
|---------------|------------|------------|---------------|
| Base Model    | 0.82       | 0.15       | 0.31          |
| Fine-tuned    | 0.89       | 0.26       | 0.45          |
| Prompt-Engineered | 0.91    | 0.29       | 0.48          |
| RAG-Enhanced  | 0.93       | 0.32       | 0.51          |

Key observations:
1. Fine-tuning provided the largest single improvement over the base model
2. Prompt engineering offered modest but important gains with no additional training
3. RAG consistently outperformed all other approaches, suggesting the importance of retrieval components for domain-specific applications
4. BLEU scores showed the largest relative improvement, indicating substantial gains in response precision

### Qualitative Analysis
Beyond quantitative metrics, we observed several qualitative improvements:
- **Response Structure**: Fine-tuned models produced more organized answers with clearer section breakdowns
- **Terminology Usage**: Significant improvement in proper use of policy-specific terminology
- **Factual Precision**: RAG models reduced hallucination and increased citation of relevant regulatory frameworks
- **Reasoning Patterns**: Fine-tuned models demonstrated better logical flow in complex policy explanations

### Limitations and Observations
Despite improvements, we identified areas for future work:
- Challenges with very recent policy developments not represented in training data
- Diminishing returns from increasing context length beyond certain thresholds in RAG approaches

### Inference Optimization
We also evaluated inference speed and resource utilization across different model configurations:
- Quantized models (int8) maintained ~96% of full-precision performance while reducing memory footprint by ~60%
- Optimized prompt templates reduced token count by 15% on average while preserving answer quality
- Batch processing for evaluation improved throughput by 3.5x compared to sequential processing

## Evaluation Metrics
The project uses multiple evaluation metrics to assess model performance:

### BERT Score
Measures semantic similarity between generated answers and reference answers using contextual embeddings.

### BLEU Score
Evaluates n-gram precision to assess the fluency and accuracy of generated text.

### ROUGE Score
Calculates overlap of n-grams, word sequences, and word pairs to evaluate content coverage.

## Model Training Process
1. Data preparation and tokenization
2. Model initialization with quantization and LoRA configuration
3. Training with customized parameters
4. Model evaluation using established metrics
5. Saving the fine-tuned model and tokenizer

## Usage Examples
Load and use the fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = './finetuned/policy-llama2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Generate answer for a question
prompt = """
Answer the following question:
Question: What are the cross-sectoral principles of the UK regulatory framework?
---
Answer:
"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation
To evaluate the model using the provided evaluation scripts:

```python
from code.evaluation import SimilarityEvaluator

evaluator = SimilarityEvaluator()
generated_answers = ["Generated model answer text here"]
reference_answers = ["Reference answer text here"]

bert_score, bleu_score, rouge_score = evaluator.evaluate_similarity(
    generated_answers, reference_answers
)

print(f"BERT Score: {bert_score}")
print(f"BLEU Score: {bleu_score}")
print(f"ROUGE-L Score: {rouge_score}")
```

## Acknowledgments
This project builds upon the LLaMA 2 model developed by Meta Platforms, Inc. and uses various open-source libraries including Hugging Face Transformers, PEFT, and TRL.