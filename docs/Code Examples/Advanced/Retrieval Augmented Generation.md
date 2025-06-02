# Retrieval Augmented Generation

Retrieval Augmented Generation (RAG) represents a significant leap forward in AI system design, combining the creative power of language models with precise information retrieval capabilities. This tutorial will guide you through building sophisticated RAG systems using Synalinks, moving from basic document retrieval to advanced Knowledge Augmented Generation (KAG) architectures.

###  Understanding the Foundation

RAG systems solve a fundamental limitation of traditional language models: their reliance on static training data. While language models excel at generating coherent text, they cannot access information beyond their training cutoff or incorporate real-time data. RAG bridges this gap by dynamically retrieving relevant information and weaving it into the generation process.

The architecture follows three core stages. The retrieval stage searches through external knowledge bases to find relevant documents or knowledge fragments. The augmentation stage enhances the original query with this retrieved context, providing the language model with additional information to work with. Finally, the generation stage produces responses that synthesize both the user's query and the retrieved knowledge.

Synalinks streamlines this complex process through its modular architecture, allowing you to compose retrieval and generation components with precision while maintaining flexibility for different use cases.

## Understanding RAG Architecture

Synalinks streamlines RAG implementation through its modular architecture, allowing you to compose retrieval and generation components with precision and flexibility.

The foundation of any RAG system begins with defining your data models. These models structure how information flows through your pipeline and ensure consistency across components. You check the tutorial about knowledge graph schemas to have a description of each data model.

```python
import synalinks
import asyncio
from typing import Literal
from typing import Union

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The answer to the user query",
    )

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/gemma",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large",
    )

    knowledge_base = synalinks.KnowledgeBase(
        index_name="neo4j://localhost:7687",
        entity_models=[City, Country, Place, Event],
        relation_models=[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=False,
    )
    
    inputs = synalinks.Input(data_model=Query)
    query_result = await synalinks.EntityRetriever(
        entity_models=[City, Country, Place, Event],
        knowledge_base=knowledge_base,
        language_model=language_model,
        return_inputs=True,
        return_query=True,
    )(inputs)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        instructions=[
            "Your task is to answer in natural language to the query based on the results of the search",
            "If the result of the search is not relevant, just say that you don't know",
        ],
        return_inputs=True,
    )(query_result)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_rag",
        description="A simple RAG program",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge/retrieval",
        show_trainable=True,
        show_schemas=True,
    )
    
    result = await program(Query(query="What is the French capital?"))
    
    print(result.prettify_json())
    

if __name__ == "__main__":
    asyncio.run(main())
```

![simple_rag](../../assets/simple_rag.png)

The `Query` and `Answer` data models serve as the input and output contracts for your RAG system. The `Query` model captures user questions, while the `Answer` model structures the system's responses. This explicit modeling ensures type safety and makes your pipeline's behavior predictable.

The language model uses Ollama's Gemma model. The embedding model, `mxbai-embed-large`, transforms text into numerical vectors that enable semantic similarity calculations during retrieval.

The knowledge base represents the heart of your RAG system. By connecting to a Neo4j graph database, it stores and indexes your knowledge using both entity models (City, Country, Place, Event) and relationship models (IsCapitalOf, IsLocatedIn). The cosine metric ensures that semantically similar content receives higher relevance scores during retrieval.

The `EntityRetriever` component searches through your knowledge base to find entities that match the user's query. It returns both the retrieved entities and maintains the original input for downstream processing. The `Generator` then combines the retrieved context with the original query to produce natural language answers.

### Advanced Knowledge Augmented Generation

Moving beyond simple entity retrieval, Knowledge Augmented Generation (KAG) architectures unlock more sophisticated reasoning capabilities by leveraging the relationships between entities in your knowledge graph.

```python
import synalinks
import asyncio
from typing import Literal
from typing import Union

from knowledge_graph_schema import City, Country, Place, Event
from knowledge_graph_schema import IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The answer to the user query",
    )

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/gemma",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large",
    )

    knowledge_base = synalinks.KnowledgeBase(
        index_name="neo4j://localhost:7687",
        entity_models=[City, Country, Place, Event],
        relation_models=[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=False,
    )
    
    inputs = synalinks.Input(data_model=Query)
    query_result = await synalinks.KnowledgeRetriever(
        entity_models=[City, Country, Place, Event],
        relation_models=[IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn],
        knowledge_base=knowledge_base,
        language_model=language_model,
        return_inputs=True,
        return_query=True,
    )(inputs)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        instructions=[
            "Your task is to answer in natural language to the query based on the results of the search",
            "If the result of the search is not relevant, just say that you don't know",
        ],
        return_inputs=True,
    )(query_result)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_kag",
        description="A simple KAG program",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/knowledge/retrieval",
        show_trainable=True,
        show_schemas=True,
    )
    
    result = await program(Query(query="What is the French capital?"))
    
    print(result.prettify_json())
    

if __name__ == "__main__":
    asyncio.run(main())
```

The key difference between the basic RAG and KAG approaches lies in the retriever component. While `EntityRetriever` focuses on finding individual entities, `KnowledgeRetriever` explores the rich web of relationships between entities. This enables more sophisticated reasoning patterns.

When you ask "What is the French capital?", the KAG system doesn't just find entities related to France or capitals. It traverses the IsCapitalOf relationships to understand the specific connection between Paris and France, providing more accurate and contextually rich answers.

The relationship models (IsCapitalOf, IsLocatedIn, IsCityOf, TookPlaceIn) define the types of connections your system can reason about. This structured approach enables complex queries like "What events took place in cities that are capitals of European countries?" by following chains of relationships across your knowledge graph.

The `return_inputs=True` parameter in both retriever and generator components ensures that information flows through your pipeline without loss. This allows downstream components to access both the original query and any intermediate results, enabling more sophisticated processing strategies.
The instruction set for the generator provides crucial guidance for response generation. The instruction to acknowledge when search results aren't relevant prevents hallucination and maintains system reliability. You can customize these instructions based on your specific use case requirements.

Don't forget that these instructions can be optimized with Synalinks to enhance the reasoning capabilities of your RAGs.

## Key Takeaways

- **Dynamic Knowledge Integration**: RAG systems bridge the gap between static training data and real-time information needs by dynamically retrieving and incorporating external knowledge. This enables AI systems to provide current, accurate responses without requiring model retraining.

- **Three-Stage Architecture**: The retrieval-augmentation-generation pipeline creates a clear separation of concerns where each stage can be optimized independently. This modular approach improves maintainability and allows for targeted performance improvements.

- **Entity vs Relationship Retrieval**: EntityRetriever focuses on finding individual knowledge components, while KnowledgeRetriever explores the rich web of relationships between entities. This distinction enables different reasoning patterns depending on query complexity.

- **Schema-Driven Pipeline Design**: Synalinks enforces structured data flow through explicit Query and Answer models, ensuring type safety and predictable behavior across your entire RAG pipeline. This contract-based approach prevents data inconsistencies and enables reliable processing.

- **Graph-Based Knowledge Representation**: Using Neo4j with defined entity and relationship models creates a structured knowledge foundation that supports both simple lookups and complex traversal queries. This approach scales from basic Q&A to sophisticated reasoning tasks.

- **Flexible Component Composition**: The modular architecture allows you to compose retrieval and generation components with precision while maintaining flexibility for different use cases. Components can be swapped, optimized, or extended without affecting the entire pipeline.