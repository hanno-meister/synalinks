# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

"""
We provide different backend-dependent `DataModel`s to use.

These data models provide basic functionality for GraphRAGs, Agents etc.

The user can build new data models by inheriting from these base models.

The checking functions works for every type of data models,
e.g. `SymbolicDataModel`, `JsonDataModel`, `DataModel` or `Variable`.
"""

import uuid
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export(
    [
        "synalinks.backend.GenericOutputs",
        "synalinks.GenericOutputs",
    ]
)
class GenericOutputs(DataModel):
    """A generic outputs"""

    outputs: Dict[str, Any]


@synalinks_export(
    [
        "synalinks.backend.GenericInputs",
        "synalinks.GenericInputs",
    ]
)
class GenericInputs(DataModel):
    """A generic inputs"""

    inputs: Dict[str, Any]


@synalinks_export(
    [
        "synalinks.backend.GenericIO",
        "synalinks.GenericIO",
    ]
)
class GenericIO(DataModel):
    """A pair of generic inputs/outputs"""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


@synalinks_export(
    [
        "synalinks.backend.ChatRole",
        "synalinks.ChatRole",
    ]
)
class ChatRole(str, Enum):
    """The chat message roles"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@synalinks_export(
    [
        "synalinks.backend.ChatMessage",
        "synalinks.ChatMessage",
    ]
)
class ChatMessage(DataModel):
    """A chat message"""

    role: ChatRole
    content: str


@synalinks_export(
    [
        "synalinks.backend.is_chat_message",
        "synalinks.is_chat_message",
    ]
)
def is_chat_message(x):
    """Checks if the given data model is a chat message

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ChatMessage.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.ChatMessages",
        "synalinks.ChatMessages",
    ]
)
class ChatMessages(DataModel):
    """A list of chat messages"""

    messages: List[ChatMessage] = []


@synalinks_export(
    [
        "synalinks.backend.is_chat_messages",
        "synalinks.is_chat_messages",
    ]
)
def is_chat_messages(x):
    """Checks if the given data model are chat messages

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ChatMessages.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Embedding",
        "synalinks.Embedding",
    ]
)
class Embedding(DataModel):
    """An embedding vector"""

    embedding: List[float] = []


@synalinks_export(
    [
        "synalinks.backend.is_embedding",
        "synalinks.is_embedding",
    ]
)
def is_embedding(x):
    """Checks if the given data model is an embedding

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Embedding.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Embeddings",
        "synalinks.Embeddings",
    ]
)
class Embeddings(DataModel):
    """A list of embeddings"""

    embeddings: List[List[float]] = []


@synalinks_export(
    [
        "synalinks.backend.is_embeddings",
        "synalinks.is_embeddings",
    ]
)
def is_embeddings(x):
    """Checks if the given data model are embeddings

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Embeddings.get_schema()):
        return True
    return False


@synalinks_export("synalinks.backend.Unique")
class Unique(DataModel):
    """A unique data model"""

    uuid: str = str(uuid.uuid4())


@synalinks_export("synalinks.backend.Weight")
class Weight(DataModel):
    """A weighted data model"""

    weight: float = 1.0


@synalinks_export("synalinks.backend.Reward")
class Reward(DataModel):
    """A rewarded data model"""

    reward: float = 0.0
    count: int = 0


@synalinks_export(
    [
        "synalinks.backend.Entity",
        "synalinks.Entity",
    ]
)
class Entity(Unique, Embeddings):
    """An entity data model"""

    pass


@synalinks_export(
    [
        "synalinks.backend.is_entity",
        "synalinks.is_entity",
    ]
)
def is_entity(x):
    """Checks if the given data model is an entity

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Entity.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Entities",
        "synalinks.Entities",
    ]
)
class Entities(DataModel):
    """A list of entities"""

    entities: List[Entity] = []


@synalinks_export(
    [
        "synalinks.backend.is_entities",
        "synalinks.is_entities",
    ]
)
def is_entities(x):
    """Checks if the given data model are entities

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Entities.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Document",
        "synalinks.Document",
    ]
)
class Document(Entity):
    """A document"""

    text: str


@synalinks_export(
    [
        "synalinks.backend.is_document",
        "synalinks.is_document",
    ]
)
def is_document(x):
    """Checks if the given data model is a document

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Document.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Prediction",
        "synalinks.Prediction",
    ]
)
class Prediction(Entity, GenericIO):
    """The generator's prediction"""

    reward: Optional[float] = None  # None if not yet backpropagated


@synalinks_export(
    [
        "synalinks.backend.is_prediction",
        "synalinks.is_prediction",
    ]
)
def is_prediction(x):
    """Checks if the given data model is a prediction

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Prediction.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Instructions",
        "synalinks.Instructions",
    ]
)
class Instructions(Entity):
    """The generator's instructions"""

    instructions: List[str]
    reward: Optional[float] = None  # None if not yet backpropagated


@synalinks_export(
    [
        "synalinks.backend.is_instructions",
        "synalinks.is_instructions",
    ]
)
def is_instructions(x):
    """Checks if the given data model is a instructions

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Instructions.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Edge",
        "synalinks.Edge",
    ]
)
class Edge(Entity, Weight):
    """An edge entity"""

    source: str
    target: str


@synalinks_export(
    [
        "synalinks.backend.is_edge",
        "synalinks.is_edge",
    ]
)
def is_edge(x):
    """Checks if the given data model is an edge

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Edge.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.KnowledgeGraph",
        "synalinks.KnowledgeGraph",
    ]
)
class KnowledgeGraph(DataModel):
    """A knowledge graph data model"""

    nodes: List[Entity] = []
    edges: List[Edge] = []


@synalinks_export(
    [
        "synalinks.backend.is_knowledge_graph",
        "synalinks.is_knowledge_graph",
    ]
)
def is_knowledge_graph(x):
    """Checks if the given data model is a knowledge graph

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), KnowledgeGraph.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.KnowledgeGraphs",
        "synalinks.KnowledgeGraphs",
    ]
)
class KnowledgeGraphs(DataModel):
    """A list of knowledge graphs"""

    knowledge_graphs: List[KnowledgeGraph] = []


@synalinks_export(
    [
        "synalinks.backend.is_knowledge_graphs",
        "synalinks.is_knowledge_graphs",
    ]
)
def is_knowledge_graphs(x):
    """Checks if the given data model are knowledge graphs

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), KnowledgeGraphs.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.TripletSearch",
        "synalinks.TripletSearch",
    ]
)
class TripletSearch(DataModel):
    subject_label: str = Field(
        description="The label of the subject node",
    )
    subject_similarity_search: str = Field(
        description=(
            "The similarity queries to match specific subject nodes"
            " (use `*` to match them all)",
        ),
    )
    return_subject: bool = Field(
        description="If True, returns the objects of the triplet search",
    )
    where_not: bool = Field(
        description="If True, check for the absence of a relation",
    )
    relation_label: str = Field(
        description="The label of the relation",
    )
    object_label: str = Field(
        description="The label of the object node",
    )
    object_similarity_search: str = Field(
        description=(
            "The similarity queries to match specific object nodes"
            " (use `*` to match them all)"
        ),
    )
    return_object: bool = Field(
        description="If True, returns the objects of the triplet search",
    )
    returns_count: bool = Field(
        description="If True, returns the count",
    )


@synalinks_export(
    [
        "synalinks.backend.RelationSchema",
        "synalinks.RelationSchema",
    ]
)
class RelationSchema(DataModel):
    relation: str = Field(description="The label of the relation")
    subject_labels: List[str] = Field(
        description="The list of entity labels that can be subject of the relation"
    )
    object_labels: List[str] = Field(
        description="The list of entity labels that can be object of the relation"
    )
    inverse_of: Optional[str] = Field(
        default=None,
        description="The inverse of the relation if any",
    )
    is_transitive: bool = Field(
        default=False, description="True if the relation is transitive"
    )
    is_symmetric: bool = Field(
        default=False, description="True if the relation is symmetric"
    )
    is_reflexive: bool = Field(
        default=False, description="True if the relation is reflexive"
    )
    is_irreflexive: bool = Field(
        default=False, description="True if the relation is irreflexive"
    )
