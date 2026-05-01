## Separated entity and relation extraction for GraphRAG
- This repository contains codes to test separating entity and relation extraction for GraphRAG
- In standard GraphRAG, entities and relations are extracted together in a single prompt to LLM
- However, for certain types of documents such as novels, especially Chinese novels, LLM has no good idea about what entities to extract, resulting in a noisy graph.
- We tested the idea of finetuning a small LLM (Qwen3.5-9B) to specialize in entity extraction, then prompt LLM to extract relation triples based on entities extracted by the finetuned model.

## GraphRAG backbone
- We use Tencent's youtu-GraphRAG with some customization and modifications.
- Major change:
    1. Removed schema evolution and switched to async prompting to significantly speed up graph generation (improve from more than 10hrs to about 30 minutes)
    2. Fixed a few bugs in LLM response parsing to eliminate errors and unexpected behaviors in experiment
    3. Modified the graph generation logic so that it can take a list of entities as input and extract relation triples based on input entities.
- Instructions to run GraphRAG:
    1. Configure LLM models and keys in .env
    2. Set up configs as a yaml file
    3. Run in CLI: python main.py --datasets [name of dataset] --config [yaml config file] 
 
## Finetune Entity Extraction
- Finetuned Qwen3.5-9B on entity extraction with three datasets: fiNERweb, CLUENER, MSRA.
- Extract entities from three dataset:
    1. Anony_chs: Publicly available dataset of popular antient Chinese novels with main entities anonymized.
    2. Novel 1: A popular Chinese online novel written in 2025
    3. Novel 2: A popular Chinese online novel written in 2010
 
