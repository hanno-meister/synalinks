# Knowledge Extraction

Knowledge extraction from unstructured data is a cornerstone of neuro-symbolic AI applications, enabling systems to transform raw text into structured, logically queryable information. Synalinks provides a sophisticated framework that supports constrained property graph extraction and querying, offering unprecedented flexibility in how you architect your knowledge extraction pipelines.

Synalinks leverages constrained property graphs as its foundation, where the schema is rigorously enforced through constrained JSON decoding. This approach ensures data integrity while maintaining the flexibility to store extracted knowledge in dedicated graph databases for efficient querying and retrieval.
The framework's modular design allows you to compose extraction pipelines from discrete, reusable components. Each component can be optimized independently, tested in isolation, and combined with others to create sophisticated data processing workflows.

```python
import synalinks
import asyncio
from typing import Literal
from typing import List
from typing import Union

# We start by defining our input data, in that case a simple Document

class Document(synalinks.DataModel):
    filename: str
    content: str

```

For the graph schema, we are going to use the same as presented in the previous lesson.

## Modular Architecture with Granular Control

The true strength of Synalinks emerges from its modular approach to pipeline composition. This foundation allows you to modularize the granularity of your data pipelines and recombine them with precision, adapting to the varying computational requirements of different extraction tasks.

In production environments, data models often exhibit vastly different inference complexities. Some entities require sophisticated reasoning and substantial computational resources to extract accurately, while others can be identified through lightweight pattern matching. A rigid, one-size-fits-all approach typically leads to suboptimal resource utilization and compromised accuracy.

Synalinks addresses this challenge by enabling you to decompose complex extraction tasks into specialized stages, each optimized for its specific requirements. This granular control allows you to allocate computational resources where they're most needed while maintaining overall pipeline efficiency.

### One-Stage Extraction

For scenarios where you have access to powerful language models capable of handling complex, multi-faceted extraction tasks, the one-stage approach offers simplicity and directness. This method excels when working with large proprietary models that possess the capacity to simultaneously identify entities, infer relationships, and maintain semantic coherence across the entire knowledge graph.

```python

# We group the entities and relations in a knowledge graph

class Knowledge(synalinks.KnowledgeGraph):
    entities: List[Union[City, Country, Place, Event]]
    relations: List[Union[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn]]


async def one_stage_program(
    language_model: synalinks.LanguageModel,
    knowledge_base: synalinks.KnowledgeBase,
):
    inputs = synalinks.Input(data_model=Document)
    knowledge_graph = await synalinks.Generator(
        data_model=Knowledge,
        language_model=language_model,
    )(inputs)
    embedded_knowledge_graph = await synalinks.Embedding(
        embedding_model=embedding_model,
    )(knowledge_graph)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="two_stage_extraction",
        description="A two stage KG extraction pipeline",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge_extraction",
        show_trainable=True,
    )

    return program

```

![one_stage_extraction](../../assets/one_stage_extraction.png)

The one-stage approach minimizes latency and reduces the complexity of pipeline orchestration. However, it demands models with substantial reasoning capabilities and may not be effective for scenarios involving smaller, specialized models.

### Two-Stage Extraction

The two-stage approach represents a strategic decomposition of the extraction process, separating entity identification from relationship inference. This separation allows for specialized optimization at each stage and provides greater control.

```python

class MapEntities(synalinks.Entities):
    entities: List[Union[City, Country, Place, Event]]

class MapRelations(synalinks.Relations):
    relations: List[Union[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn]]


async def two_stage_program(
    language_model: synalinks.LanguageModel,
    knowledge_base: synalinks.KnowledgeBase,
):
    inputs = synalinks.Input(data_model=Document)
    entities = await synalinks.Generator(
        data_model=MapEntities,
        language_model=language_model,
    )(inputs)
    
    # inputs_with_entities = inputs AND entities (See Control Flow tutorial)
    inputs_with_entities = inputs & entities 
    relations = await synalinks.Generator(
        data_model=MapRelations,
        language_model=language_model,
    )(inputs_with_entities)
    
    # knowledge_graph = inputs AND entities (See Control Flow tutorial)
    knowledge_graph = entities & relations

    embedded_knowledge_graph = await synalinks.Embedding(
        embedding_model=embedding_model,
    )(knowledge_graph)

    updated_knowledge_graph = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded_knowledge_graph)

    outputs = updated_knowledge_graph

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="two_stage_extraction",
        description="A two stage KG extraction pipeline",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge_extraction",
        show_trainable=True,
    )

    return program

```

![two_stage_extraction](../../assets/two_stage_extraction.png)

This staged approach offers several advantages: entities can be extracted using lightweight models optimized for named entity recognition, while relationship inference can leverage more sophisticated reasoning models.

### Multi-Stage Extraction

Complex real-world applications often require extraction of diverse data types with varying computational demands. Multi-stage pipelines excel in scenarios where you need to simultaneously extract factual entities, perform complex reasoning, and generate analytical insights from the same source material.

Consider a scenario where you need to extract both concrete geographical entities and abstract reasoning patterns from documents. The multi-stage approach allows you to optimize each extraction task independently while maintaining coherent integration of the results. You might employ fast pattern-matching models for entity extraction while reserving computationally intensive reasoning models for inference tasks, all within a single, coordinated pipeline.

This flexibility becomes crucial when dealing with heterogeneous data sources, varying quality requirements, or when different stages require different specialized models. Multi-stage pipelines also enable sophisticated error handling and recovery strategies, where failures in one stage don't necessarily compromise the entire extraction process.

The modular architecture of Synalinks ensures that as your extraction requirements evolve, you can incrementally enhance your pipelines without rebuilding them from scratch. This evolutionary approach to knowledge extraction provides the adaptability needed for production systems that must handle changing data patterns and evolving business requirements.

```python

async def multi_stage_program(
    language_model: synalinks.LanguageModel,
    embedding_model: synalinks.EmbeddingModel,
    knowledge_base: synalinks.KnowledgeBase,
):
    inputs = synalinks.Input(data_model=Document)
    entities = await synalinks.Generator(
        data_model=KnowledgeEntities,
        language_model=language_model,
    )(inputs)
    notes = await synalinks.Generator(
        data_model=Notes,
        language_model=language_model,
    )(inputs)

    # inputs_with_entities = inputs AND entities (See Control Flow tutorial)
    inputs_with_entities = inputs & entities
    relations = await synalinks.Generator(
        data_model=KnowledgeRelations,
        language_model=language_model,
    )(inputs_with_entities)

    # knowledge_graph = inputs AND entities (See Control Flow tutorial)
    knowledge_graph = entities & relations

    embedded_knowledge_graph = await synalinks.Embedding(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(knowledge_graph)

    embedded_notes = await synalinks.Embedding(
        embedding_model=embedding_model,
        in_mask=["description"],
    )(notes)

    updated_knowledge_graph = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded_knowledge_graph)
    
    updated_notes = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded_notes)

    outputs = [updated_knowledge_graph, updated_notes]

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="multi_stage_extraction",
        description="A multi stage KG extraction pipeline",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge_extraction",
        show_trainable=True,
    )

    return program
```

![multi_stage_extraction](../../assets/multi_stage_extraction.png)

```python
async def main():
    
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large",
    )
    
    knowledge_base = synalinks.KnowledgeBase(
        index_name="neo4j://localhost:7687",
        entity_models=[City, Country, Place, Event, Notes],
        relation_models=[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=False,
    )
    
    program = await one_stage_program(
        language_model=language_model,
        embedding_model=embedding_model,
        knowledge_base=knowledge_base,
    )
    
    program_1 = await two_stage_program(
        language_model=language_model,
        embedding_model=embedding_model,
        knowledge_base=knowledge_base,
    )
    
    program_2 = await multi_stage_program(
        language_model=language_model,
        embedding_model=embedding_model,
        knowledge_base=knowledge_base,
    )
```

## Conclusion

Synalinks represents a paradigm shift in knowledge extraction, moving beyond monolithic, inflexible approaches toward a modular, production-first framework that adapts to the complexities of real-world applications.

The modular architecture enables teams to iteratively refine their extraction pipelines, starting with simple one-stage approaches and evolving toward sophisticated multi-stage systems as requirements become more complex. This evolutionary path reduces implementation risk while providing a clear migration strategy for growing applications.

By separating concerns across different stages, Synalinks allows for specialized optimization at each level of the extraction hierarchy. Entity extraction can leverage lightweight, fast models, while complex reasoning tasks can utilize more powerful, specialized models. This granular control over computational resources leads to more efficient systems that deliver better results at lower costs.

Whether you're building a simple entity extraction system or a comprehensive knowledge discovery platform, Synalinks provides the flexibility, control, and scalability needed to transform unstructured data into actionable insights. 

In an era where structured data is the new oil, Synalinks provides the refinery that transforms raw information into structured knowledge, enabling organizations to unlock the full potential of their data assets through intelligent, adaptive extraction pipelines.

### Key Takeaways

- **Schema-First Design**: Synalinks enforces structured schemas through constrained JSON decoding, ensuring data integrity and consistency across your entire knowledge extraction pipeline. This contract-based approach prevents schema drift and enables reliable downstream processing.
- **Logical Flow Composition**: The framework's mathematical foundation allows for precise pipeline composition using logical operations. This enables sophisticated data flow patterns where outputs from one stage can be combined with inputs using python logical operators, creating complex but maintainable extraction workflows.
- **Modular Components**: Each pipeline component can be developed, tested, and optimized independently. This separation of concerns reduces complexity, improves maintainability, and enables teams to specialize in different aspects of the extraction process.
- **One-Stage for Simplicity**: Use single-stage extraction when you have access to powerful models capable of handling comprehensive extraction tasks. This approach minimizes latency and orchestration complexity but requires models with substantial reasoning capabilities.
- **Two-Stage for Balance**: Implement two-stage pipelines when you need to balance accuracy with computational efficiency. This approach allows specialized optimization for entity extraction and relationship inference while maintaining manageable complexity.
- **Multi-Stage for Sophistication**: Deploy multi-stage architectures for complex scenarios requiring diverse extraction types, specialized models for different tasks, and sophisticated reasoning capabilities. This approach maximizes flexibility and performance optimization opportunities.
- **Resource Optimization**: Different stages can utilize different models optimized for their specific tasks, leading to better resource utilization and cost efficiency. Lightweight models handle simple tasks while powerful models focus on complex reasoning.
- **Error Isolation**: Failures in one stage don't necessarily compromise the entire pipeline depending on the logical operators used. This resilience is crucial for production systems processing large volumes of heterogeneous data.