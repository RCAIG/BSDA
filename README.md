# Multi-Agent Disaster Damage Assessment

This is a multi-agent pipeline for disaster damage assessment from paired pre-disaster and post-disaster visual observations. The system combines visual perception, damage change detection, severity assessment, critic-driven revision, retrieval-augmented generation, and optional web search to produce structured damage analysis outputs.

The project is designed around modular agents so that each stage can be developed, tested, and replaced independently.

## Overview

The pipeline follows four main stages:

1. **Perception Agent** converts image evidence into structured scene descriptions.
2. **Detection Agent** compares pre-disaster and post-disaster descriptions to identify candidate damage-related changes.
3. **Assessment Agent** assigns severity labels and supporting evidence to detected changes.
4. **Critic Agent** reviews the assessment output, flags inconsistencies, and can drive revision rounds.

The orchestration layer can run these stages as a graph-based workflow and supports iterative revision through critic feedback.


## Repository Structure

```text
.
├── Agents/
│   ├── PerceptionAgent/      # Image-to-description perception logic
│   ├── DetectionAgent/       # Change extraction and damage detection
│   ├── AssessmentAgent/      # Severity assessment and evidence reasoning
│   ├── CriticAgent/          # Output review and revision guidance
│   ├── tools/                # RAG tools, web search tools, registry, ReAct loop
│   ├── pipeline.py           # High-level pipeline utilities
│   └── shared_llm.py         # Shared local model loading utilities
├── orchestrator/
│   └── langgraph/            # Graph workflow and node definitions
├── schemas/                  # Shared pipeline state schemas
├── src/                      # Additional agent abstractions and utilities
├── rag_query_local.py        # Local RAG query helper
├── requirements.txt          # Project dependencies
└── README.md
```

## Agent Modules

### Perception Agent

The Perception Agent prepares structured descriptions from visual inputs. These descriptions are used by downstream agents instead of raw images when performing change detection and severity reasoning.

Main files:

- `Agents/PerceptionAgent/main.py`
- `Agents/PerceptionAgent/perception_agent_agentic.py`

### Detection Agent

The Detection Agent compares pre-disaster and post-disaster descriptions from the same location. It extracts candidate changes, filters likely non-disaster differences, and attaches retrieval evidence when available.

Main files:

- `Agents/DetectionAgent/detection_agent.py`
- `Agents/DetectionAgent/detection_agent_ragws.py`
- `Agents/DetectionAgent/rag.py`
- `Agents/DetectionAgent/config.py`

### Assessment Agent

The Assessment Agent evaluates detected changes and assigns severity labels with evidence. It can query rule-oriented and case-oriented retrieval sources.

Main files:

- `Agents/AssessmentAgent/assessment_agent.py`
- `Agents/AssessmentAgent/assessment_agent_ragws.py`
- `Agents/AssessmentAgent/rag.py`
- `Agents/AssessmentAgent/config.py`

### Critic Agent

The Critic Agent reviews outputs for inconsistency, weak evidence, grading mismatch, or revision needs. It supports critic-driven iteration through the orchestration layer.

Main files:

- `Agents/CriticAgent/critic_agent.py`
- `Agents/CriticAgent/critic_agent_ragws.py`
- `Agents/CriticAgent/revision_orchestrator.py`
- `Agents/CriticAgent/rag.py`

## RAG and Web Search Tools

The project keeps retrieval and web search support under `Agents/tools/`.

Main tool files:

- `Agents/tools/rag_tools.py`: standard tool wrappers for internal RAG, government-rule retrieval, historical-case retrieval, and event context lookup.
- `Agents/tools/rag_ws.py`: RAG with web search fallback, query rewriting, and evidence sufficiency evaluation.
- `Agents/tools/ddgs_search.py`: web search tools for general search, real-time disaster information, and supplemental disaster context.
- `Agents/tools/registry.py`: shared tool registry.
- `Agents/tools/react_loop.py`: ReAct-style tool execution loop and agent-specific tool registration.

Tool registration is organized by agent:

- Detection: internal RAG, web search, real-time disaster search, and event context.
- Assessment: rule RAG, history-case RAG, and web search.
- Critic: rule RAG and web search.

## Orchestration

The LangGraph workflow is defined under `orchestrator/langgraph/`.

Important files:

- `orchestrator/langgraph/graph.py`: builds the workflow graph.
- `orchestrator/langgraph/run_revision.py`: command-line entry point for revision workflows.
- `orchestrator/langgraph/nodes/`: graph nodes for initialization, perception, detection, assessment, critique, revision decisions, and final output.
- `schemas/state.py`: shared graph state definition.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For Windows users, FAISS may be easier to install through conda:

```bash
conda install -c conda-forge faiss-cpu
```

## Configuration

Before running the pipeline, configure local paths in the relevant config files:

- `Agents/DetectionAgent/config.py`
- `Agents/AssessmentAgent/config.py`
- `Agents/CriticAgent/config.py`

Typical values to configure:

- local model path
- RAG artifacts directory
- pre-disaster description directory
- post-disaster description directory
- output directory
- retrieval corpus and top-k values

The public code uses generic placeholder paths. Replace them with paths that match your local environment.

## Usage

### Run the LangGraph Revision Pipeline

```bash
python -m orchestrator.langgraph.run_revision \
  --pre_path path/to/pre_description.json \
  --post_path path/to/post_description.json \
  --pair_id sample_pair \
  --max_revisions 1 \
  --use_rag 1 \
  --use_llm 1 \
  --output_dir outputs
```

Optional image paths can be provided if perception revision is enabled:

```bash
python -m orchestrator.langgraph.run_revision \
  --pre_path path/to/pre_description.json \
  --post_path path/to/post_description.json \
  --pre_image_path path/to/pre_image.jpg \
  --post_image_path path/to/post_image.jpg \
  --pair_id sample_pair \
  --max_revisions 1
```

### Use Individual Agents

Each agent can also be imported and used independently from Python code. This is useful for debugging one stage at a time or replacing a module with a custom implementation.

Example:

```python
from Agents.DetectionAgent.detection_agent import DetectionAgent
from Agents.DetectionAgent.rag import DamageFeatureRAG
```

## Output Format

The pipeline is designed to produce structured JSON-style outputs that can include:

- pair identifier
- detected changes
- evidence snippets
- damage type
- severity or grade
- confidence
- RAG evidence
- critic judgement
- revision recommendation

Exact fields may vary by agent and workflow configuration.

## Development Notes

- Keep agent interfaces structured and JSON-friendly.
- Prefer adding new tools through `Agents/tools/registry.py`.
- Keep model and data paths configurable rather than hard-coded.
- Use RAG evidence as supporting context, not as a replacement for visible evidence.
- Use the Critic Agent for consistency checking and revision control.

## License

Add a license file before public release if this repository will be distributed or reused by others.
