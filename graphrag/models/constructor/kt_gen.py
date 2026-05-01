import json
import os
import asyncio
import time
from tenacity import retry, wait_fixed, retry_if_exception, stop_after_attempt
from typing import Any, Dict, List, Tuple

import nanoid
import networkx as nx
import tiktoken
import json_repair

from config import get_config
from utils import call_llm_api, graph_processor, tree_comm
from utils.logger import logger

class KTBuilder:
    def __init__(self, dataset_name, schema_path=None, mode=None, config=None):
        if config is None:
            config = get_config()
        
        self.config = config
        self.dataset_name = dataset_name
        self.schema = self.load_schema(schema_path or config.get_dataset_config(dataset_name).schema_path)
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.datasets_no_chunk = config.construction.datasets_no_chunk
        self.token_len = 0
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(100)
        self.llm_client = call_llm_api.LLMCompletionCall()
        self.all_chunks = {}
        self.mode = mode or config.construction.mode
        self.entities = {}
        if self.mode == "given_entity":
            self.entities = self.load_schema(config.get_dataset_config(dataset_name).entity_path)
    
    def fix_bracket_colon(self, s: str) -> str:
        """
        For each `": "` pattern, check its innermost enclosing bracket.
        If it's `[`, replace `": "` with `: ` (merging the two quoted strings).
        If it's `{`, leave it unchanged.
        """
        result = []
        i = 0
        bracket_stack = []  # tracks '[' or '{' as we go

        while i < len(s):
            if s[i] in '[{':
                bracket_stack.append(s[i])
                result.append(s[i])
                i += 1

            elif s[i] in ']}':
                bracket_stack.pop()
                result.append(s[i])
                i += 1

            # Detect `": "` — closing quote, colon, space, opening quote
            elif s[i:i+4] == '": "':
                if bracket_stack and bracket_stack[-1] == '[':
                    # Innermost bracket is [] → merge: drop `": "`, replace with `: `
                    result.append(': ')
                    i += 4
                else:
                    # Innermost bracket is {} → leave unchanged
                    result.append('": "')
                    i += 4

            else:
                result.append(s[i])
                i += 1

        return ''.join(result)
    
    def load_schema(self, schema_path) -> Dict[str, Any]:
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                return schema
        except FileNotFoundError:
            return dict()

    def _split_text_with_overlap(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_tail_tokens: int = 100,
    ) -> List[str]:
        """Split text into chunks with overlap using token count."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("gpt2")
            
        tokens = encoding.encode(text)
        if len(tokens) <= chunk_size:
            return [text]

        windows = []
        start = 0
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # Prevent infinite loop if overlap >= chunk_size

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            windows.append([start, end])
            start += step

        if len(windows) >= 2:
            idx = 0
            while idx < len(windows):
                cur_len = windows[idx][1] - windows[idx][0]
                if cur_len < min_tail_tokens:
                    if idx > 0:
                        windows[idx - 1][1] = windows[idx][1]
                        windows.pop(idx)
                        continue
                    elif idx + 1 < len(windows):
                        windows[idx + 1][0] = windows[idx][0]                                     
                        windows.pop(idx)
                        continue
                idx += 1

        chunks = []
        for s, e in windows:
            decoded_chunk = encoding.decode(tokens[s:e])
            if (e - s < 5) or (len(decoded_chunk.strip()) < 5):
                continue
            chunks.append(decoded_chunk)
        return chunks

    async def chunk_text(self, text) -> Tuple[List[str], Dict[str, str]]:
        if self.dataset_name in self.datasets_no_chunk:
            chunks = [f"{text.get('title', '')} {text.get('text', '')}".strip() 
                     if isinstance(text, dict) else str(text)]
        else:
            # Use configured chunk size and overlap
            raw_text = str(text)
            if isinstance(text, dict):
                raw_text = f"{text.get('title', '')} {text.get('text', '')}".strip()
            
            chunk_size = getattr(self.config.construction, 'chunk_size', 1000)
            overlap = getattr(self.config.construction, 'overlap', 200)
            min_tail_tokens = getattr(self.config.construction, 'min_tail_tokens', 100)
            chunks = self._split_text_with_overlap(raw_text, chunk_size, overlap, min_tail_tokens)

        chunk2id = {}
        for chunk in chunks:
            try:
                chunk_id = nanoid.generate(size=8)
                chunk2id[chunk_id] = chunk
            except Exception as e:
                logger.warning(f"Failed to generate chunk id with nanoid: {type(e).__name__}: {e}")

        async with self.lock:
            self.all_chunks.update(chunk2id)

        return chunks, chunk2id

    def _clean_text(self, text: str) -> str:
        if not text:
            return "[EMPTY_TEXT]"
        
        if self.dataset_name == "graphrag-bench":
            safe_chars = {
                *" .:,!?()-+=[]{}()\\/|_^~<>*&%$#@!;\"'`"
            }
            cleaned = "".join(
                char for char in text 
                if char.isalnum() or char.isspace() or char in safe_chars
            ).strip()
        else:
            safe_chars = {
                *" .:,!?()-+="  
            }
            cleaned = "".join(
                char for char in text 
                if char.isalnum() or char.isspace() or char in safe_chars
            ).strip()
        
        return cleaned if cleaned else "[EMPTY_AFTER_CLEANING]"
    
    def save_chunks_to_file(self):
        os.makedirs("output/chunks", exist_ok=True)
        chunk_file = f"output/chunks/{self.dataset_name}.txt"
        
        # When reconstructing, we want a fresh start, not appending to old data
        all_data = self.all_chunks
        
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in all_data.items():
                escaped = " ".join(chunk_text.splitlines()).replace('\n', '\\n').replace('\t', '\z\t')
                f.write(f"id: {chunk_id}\tChunk: {escaped}\n")
        
        logger.info(f"Chunk data saved to {chunk_file} ({len(all_data)} chunks)")
    
    @retry(
    retry=retry_if_exception(lambda e: True),  # retry on any exception
    wait=wait_fixed(60),
    stop=stop_after_attempt(5),
    reraise=True,
)
    async def extract_with_llm(self, prompt: str):
        async with self.semaphore:
            response = await self.llm_client.async_call_api(prompt)
        response = self.fix_bracket_colon(response)
        parsed_dict = json_repair.loads(response)
        parsed_json = json.dumps(parsed_dict, ensure_ascii=False)
        return parsed_json 

    def token_cal(self, text: str):
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def _get_construction_prompt(self, chunk: str) -> str:
        """Get the appropriate construction prompt based on dataset name and mode (agent/noagent)."""
        recommend_schema = json.dumps(self.schema, ensure_ascii=False)
        
        # Base prompt type mapping
        prompt_type_map = {
            "1":"novel",
            "2":"novel",
            "anony_chs": "anony_chs",
            "novel": "novel",
            "novel_eng": "novel_eng"
        }
        
        base_prompt_type = prompt_type_map.get(self.dataset_name, "novel")
        
        # Add agent suffix if in agent mode
        if self.mode == "agent":
            prompt_type = f"{base_prompt_type}_agent"
        elif self.mode == "given_entity":
            prompt_type = f"{base_prompt_type}_given_entity"
        else:
            prompt_type = base_prompt_type
        
        if self.mode == "given_entity":
            current_entities = {}
            for key in current_entities:
                if key in chunk:
                    current_entities[key] = self.entities.get(key, [])
            return self.config.get_prompt_formatted("construction", prompt_type, schema=recommend_schema, chunk=chunk,entities=current_entities)

        return self.config.get_prompt_formatted("construction", prompt_type, schema=recommend_schema, chunk=chunk)
    
    def _validate_and_parse_llm_response(self, prompt: str, llm_response: str) -> dict:
        """Validate and parse LLM response, returning None if invalid."""
        if llm_response is None:
            return None
        try:
            self.token_len += self.token_cal(prompt + llm_response)
            llm_response = self.fix_bracket_colon(llm_response)
            return json_repair.loads(llm_response)
        except Exception as e:
            llm_response_str = str(llm_response) if llm_response is not None else "None"
            logger.error(f"Failed to parse LLM response: {type(e).__name__}: {e}. Response content: {llm_response_str}")
            return None
    
    def _find_or_create_entity(self, entity_name: str, chunk_id: int, nodes_to_add: list, entity_type: str = None) -> str:
        """Find existing entity or create a new one, returning the entity node ID."""
        entity_node_id = next(
            (
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("label") == "entity" and d["properties"]["name"] == entity_name
            ),
            None,
        )
        
        if not entity_node_id:
            entity_node_id = f"entity_{self.node_counter}"
            properties = {"name": entity_name, "chunk id": chunk_id}
            if entity_type:
                properties["schema_type"] = entity_type
            
            nodes_to_add.append((
                entity_node_id,
                {
                    "label": "entity", 
                    "properties": properties, 
                    "level": 2
                }
            ))
            self.node_counter += 1
                
        return entity_node_id
    
    def _validate_triple_format(self, triple: list) -> tuple:
        """Validate and normalize triple format, returning (subject, predicate, object) or None."""
        try:
            if len(triple) > 3:
                triple = triple[:3]
            elif len(triple) < 3:
                return None
            
            return tuple(triple)
        except Exception as e:
            return None
    
    def _process_attributes(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """Process extracted attributes and return nodes and edges to add."""
        nodes_to_add = []
        edges_to_add = []
        
        for entity, attributes in extracted_attr.items():
            for attr in attributes:
                # Create attribute node
                attr_node_id = f"attr_{self.node_counter}"
                nodes_to_add.append((
                    attr_node_id,
                    {
                        "label": "attribute",
                        "properties": {"name": attr, "chunk id": chunk_id},
                        "level": 1,
                    }
                ))
                self.node_counter += 1

                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self._find_or_create_entity(entity, chunk_id, nodes_to_add, entity_type)
                edges_to_add.append((entity_node_id, attr_node_id, "has_attribute"))
        
        return nodes_to_add, edges_to_add
    
    def _process_triples(self, extracted_triples: list, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """Process extracted triples and return nodes and edges to add."""
        nodes_to_add = []
        edges_to_add = []
        
        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue
                
            subj, pred, obj = validated_triple
            
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None
            
            subj_node_id = self._find_or_create_entity(subj, chunk_id, nodes_to_add, subj_type)
            obj_node_id = self._find_or_create_entity(obj, chunk_id, nodes_to_add, obj_type)
            
            edges_to_add.append((subj_node_id, obj_node_id, pred))
        
        return nodes_to_add, edges_to_add

    async def process_level1_level2(self, chunk: str, id: int):
        """Process attributes (level 1) and triples (level 2) with optimized structure."""
        prompt = self._get_construction_prompt(chunk)
        llm_response = await self.extract_with_llm(prompt)
        
        # Validate and parse response
        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)
        if not parsed_response:
            return
        
        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})
        
        # Process attributes and triples
        attr_nodes, attr_edges = self._process_attributes(extracted_attr, id, entity_types)
        triple_nodes, triple_edges = self._process_triples(extracted_triples, id, entity_types)
        
        all_nodes = attr_nodes + triple_nodes
        all_edges = attr_edges + triple_edges
        
        async with self.lock:
            for node_id, node_data in all_nodes:
                self.graph.add_node(node_id, **node_data)
            
            for u, v, relation in all_edges:
                self.graph.add_edge(u, v, relation=relation)

    def _find_or_create_entity_direct(self, entity_name: str, chunk_id: int, entity_type: str = None) -> str:
        """Find existing entity or create a new one directly in graph (for agent mode)."""
        entity_node_id = next(
            (
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("label") == "entity" and d["properties"]["name"] == entity_name
            ),
            None,
        )
        
        if not entity_node_id:
            entity_node_id = f"entity_{self.node_counter}"
            properties = {"name": entity_name, "chunk id": chunk_id}
            if entity_type:
                properties["schema_type"] = entity_type
                
            self.graph.add_node(
                entity_node_id, 
                label="entity", 
                properties=properties, 
                level=2
            )
            self.node_counter += 1
            
        return entity_node_id
    
    def _process_attributes_agent(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None):
        """Process extracted attributes in agent mode (direct graph operations)."""
        for entity, attributes in extracted_attr.items():
            for attr in attributes:
                # Create attribute node
                attr_node_id = f"attr_{self.node_counter}"
                self.graph.add_node(
                    attr_node_id,
                    label="attribute",
                    properties={
                        "name": attr,
                        "chunk id": chunk_id
                    },
                    level=1,
                )
                self.node_counter += 1

                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self._find_or_create_entity_direct(entity, chunk_id, entity_type)
                self.graph.add_edge(entity_node_id, attr_node_id, relation="has_attribute")
    
    def _process_triples_agent(self, extracted_triples: list, chunk_id: int, entity_types: dict = None):
        """Process extracted triples in agent mode (direct graph operations)."""
        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue
                
            subj, pred, obj = validated_triple
            
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None
            
            # Find or create subject and object entities
            subj_node_id = self._find_or_create_entity_direct(subj, chunk_id, subj_type)
            obj_node_id = self._find_or_create_entity_direct(obj, chunk_id, obj_type)
            
            self.graph.add_edge(subj_node_id, obj_node_id, relation=pred)

    async def process_level1_level2_agent(self, chunk: str, id: int):
        """Process attributes (level 1) and triples (level 2) with agent mechanism for schema evolution.
        
        This method enables dynamic schema evolution by allowing the LLM to suggest new entity types,
        relation types, and attribute types that can be added to the existing schema.
        """
        prompt = self._get_construction_prompt(chunk)
        llm_response = await self.extract_with_llm(prompt)

        # Validate and parse response (reuse helper method)
        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)
        if not parsed_response:
            return
        new_schema_types = parsed_response.get("new_schema_types", {})
        if new_schema_types:
            self._update_schema_with_new_types(new_schema_types)
        
        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})
        
        async with self.lock:
            self._process_attributes_agent(extracted_attr, id, entity_types)
            self._process_triples_agent(extracted_triples, id, entity_types)

    def _update_schema_with_new_types(self, new_schema_types: Dict[str, List[str]]):
        """Update the schema file with new types discovered by the agent.
        
        This method processes schema evolution suggestions from the LLM and updates
        the corresponding schema file with new node types, relations, and attributes.
        Only adds types that don't already exist in the current schema.
        
        Args:
            new_schema_types: Dictionary containing 'nodes', 'relations', and 'attributes' lists
        """
        try:
            schema_paths = {
                "hotpot": "schemas/hotpot.json",
                "2wiki": "schemas/2wiki.json", 
                "musique": "schemas/musique.json",
                "novel": "schemas/novels_chs.json",
                "graphrag-bench": "schemas/graphrag-bench.json"
            }
            
            schema_path = schema_paths.get(self.dataset_name)
            if not schema_path:
                return
                
            with open(schema_path, 'r', encoding='utf-8') as f:
                current_schema = json.load(f)
            
            updated = False
            
            if "nodes" in new_schema_types:
                for new_node in new_schema_types["nodes"]:
                    if new_node not in current_schema.get("Nodes", []):
                        current_schema.setdefault("Nodes", []).append(new_node)
                        updated = True
            
            if "relations" in new_schema_types:
                for new_relation in new_schema_types["relations"]:
                    if new_relation not in current_schema.get("Relations", []):
                        current_schema.setdefault("Relations", []).append(new_relation)
                        updated = True

            if "attributes" in new_schema_types:
                for new_attribute in new_schema_types["attributes"]:
                    if new_attribute not in current_schema.get("Attributes", []):
                        current_schema.setdefault("Attributes", []).append(new_attribute)
                        updated = True
            
            # Save updated schema back to file
            if updated:
                with open(schema_path, 'w', encoding='utf-8') as f:
                    json.dump(current_schema, f, ensure_ascii=False, indent=2)
                
                # Update the in-memory schema
                self.schema = current_schema
                
        except Exception as e:
            logger.error(f"Failed to update schema for dataset '{self.dataset_name}': {type(e).__name__}: {e}")

    def process_level4(self):
        """Process communities using Tree-Comm algorithm"""
        level2_nodes = [n for n, d in self.graph.nodes(data=True) if d['level'] == 2]
        start_comm = time.time()
        _tree_comm = tree_comm.FastTreeComm(
            self.graph, 
            embedding_model=self.config.tree_comm.embedding_model,
            struct_weight=self.config.tree_comm.struct_weight,
        )
        comm_to_nodes = _tree_comm.detect_communities(level2_nodes)

        # create super nodes (level 4 communities)
        _tree_comm.create_super_nodes_with_keywords(comm_to_nodes, level=4)
        # _tree_comm.add_keywords_to_level3(comm_to_nodes)
        # connect keywords to communities (optional)
        # self._connect_keywords_to_communities()
        end_comm = time.time()
        logger.info(f"Community Indexing Time: {end_comm - start_comm}s")
    
    def _connect_keywords_to_communities(self):
        """Connect relevant keywords to communities"""
        # comm_names = [self.graph.nodes[n]['properties']['name'] for n, d in self.graph.nodes(data=True) if d['level'] == 4]
        comm_nodes = [n for n, d in self.graph.nodes(data=True) if d['level'] == 4]
        kw_nodes = [n for n, d in self.graph.nodes(data=True) if d['label'] == 'keyword']
        for comm in comm_nodes:
            comm_name = self.graph.nodes[comm]['properties']['name'].lower()
            for kw in kw_nodes:
                kw_name = self.graph.nodes[kw]['properties']['name'].lower()
                if kw_name in comm_name or comm_name in kw_name:
                    self.graph.add_edge(kw, comm, relation="describes")

    async def process_document(self, doc):
        chunks, chunk2id = await self.chunk_text(doc)
        
        if not chunks or not chunk2id:
            raise ValueError(...)

        async def process_chunk(chunk):
            try:
                id = next(key for key, value in chunk2id.items() if value == chunk)
            except StopIteration:
                id = nanoid.generate(size=8)
                chunk2id[id] = chunk
            
            if self.mode == "agent":
                await self.process_level1_level2_agent(chunk, id)
            else:
                await self.process_level1_level2(chunk, id)

        await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

    async def process_all_documents(self, documents):
        # max_workers config key is no longer used for threading;
        # concurrency is controlled by self.semaphore (100 slots).
        start_construct = time.time()
        total_docs = len(documents)
        logger.info(f"Starting processing {total_docs} documents with semaphore limit 100...")

        processed_count = 0
        failed_count = 0

        async def _process_one(doc):
            nonlocal processed_count, failed_count
            try:
                await self.process_document(doc)
                processed_count += 1
                if processed_count % 20 == 0 or processed_count == total_docs:
                    elapsed_time = time.time() - start_construct
                    avg_time_per_doc = elapsed_time / processed_count if processed_count > 0 else 0
                    remaining_docs = total_docs - processed_count
                    estimated_remaining_time = remaining_docs * avg_time_per_doc
                    logger.info(
                        f"Progress: {processed_count}/{total_docs} documents processed "
                        f"({processed_count/total_docs*100:.1f}%) "
                        f"[{failed_count} failed] "
                        f"ETA: {estimated_remaining_time/60:.1f} minutes"
                    )
            except Exception as e:
                failed_count += 1

        await asyncio.gather(*[_process_one(doc) for doc in documents])

        end_construct = time.time()
        logger.info(f"Construction Time: {end_construct - start_construct}s")
        logger.info(f"Successfully processed: {processed_count}/{total_docs} documents")
        logger.info(f"Failed: {failed_count} documents")

        logger.info(f"🚀🚀🚀🚀 {'Processing Level 3 and 4':^20} 🚀🚀🚀🚀")
        logger.info(f"{'➖' * 20}")
        self.triple_deduplicate()
        self.process_level4()

       

    def triple_deduplicate(self):
        """deduplicate triples in lv1 and lv2"""
        new_graph = nx.MultiDiGraph()

        for node, node_data in self.graph.nodes(data=True):
            new_graph.add_node(node, **node_data)

        seen_triples = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            relation = data.get('relation') 
            if (u, v, relation) not in seen_triples:
                seen_triples.add((u, v, relation))
                new_graph.add_edge(u, v, **data)
        self.graph = new_graph

    def format_output(self) -> List[Dict[str, Any]]:
        """convert graph to specified output format"""
        output = []

        for u, v, data in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]

            relationship = {
                "start_node": {
                    "label": u_data["label"],
                    "properties": u_data["properties"],
                },
                "relation": data["relation"],
                "end_node": {
                    "label": v_data["label"],
                    "properties": v_data["properties"],
                },
            }
            output.append(relationship)

        return output
    
    def save_graphml(self, output_path: str):
        graph_processor.save_graph(self.graph, output_path)
    
    async def build_knowledge_graph(self, corpus):
        logger.info(f"========{'Start Building':^20}========")
        logger.info(f"{'➖' * 30}")
        
        with open(corpus, 'r', encoding='utf-8') as f:
            documents = json_repair.load(f)
        
        await self.process_all_documents(documents)
        
        logger.info(f"All Process finished, token cost: {self.token_len}")
        
        self.save_chunks_to_file()
        
        output = self.format_output()
        
        json_output_path = f"output/graphs/{self.dataset_name}_new.json"
        os.makedirs("output/graphs", exist_ok=True)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved to {json_output_path}")
        
        return output