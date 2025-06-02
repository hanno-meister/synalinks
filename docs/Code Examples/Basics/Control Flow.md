# Control Flow

Controlling the flow of information in a program is an essential feature of any LM framework.
In Synalinks, we implemented it in circuit-like fashion, where the flow of information can be 
conditionaly or logically restricted to only flow in a subset of a computation graph.

## Parallel Branches

To create parallel branches, all you need to do is using the same inputs when declaring the modules.
Then Synalinks will automatically detect them and run them in parrallel with asyncio.

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )

async def main():
    inputs = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    outputs = [x1, x2]

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="parallel_branches",
        description="Illustrate the use of parallel branching",
    )
    
    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

![parralel_branches](../../assets/parallel_branches.png)

## Decisions

Decisions in Synalinks can be viewed as a single label classification, they allow
the system to classify the inputs based on a question and labels to choose from.
The labels are used to create on the fly a Enum schema that ensure, thanks to
constrained structured output, that the system will answer one of the provided labels.

This module is the basis of robust control flow in Synalinks.

```python
async def main():
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Decision(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="decision_making",
        description="Illustrate the decision making process",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

![decision_making](../../assets/decision_making.png)

## Conditional Branches

To make conditional branches, we will need the help of a core module: The Branch
module. This module use a decision and route the input data model to the selected
branch. When a branch is not selected, that branch output a None.


```python
class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The correct answer",
    )

async def main():
    inputs = synalinks.Input(data_model=Query)
    (x1, x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
    )(inputs)
    outputs = [x1, x2]

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="conditional_branches",
        description="Illustrate the conditional branches",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

![conditional_branches](../../assets/conditional_branches.png)

## Data Models Operators

Synalinks implement Python operators that works with data models, some of them are
straightforward, like the concatenation, implemented in the Python `+` operator. 
But others like the `logical_and` and `logical_or` implemented respectively 
in the `&` and `|` operator are more difficult to grasp at first. As explained
above, in the conditional branches, the branch not selected will have a `None` 
as output. To account that fact and to implement logical flows, we need operators
that can work with them. See the [Ops API](https://synalinks.github.io/synalinks/Synalinks%20API/Ops%20API/)
section for an extensive list of all data model operations.

### Concatenation

The concatenation, consist in creating a data model that have the fields of both
inputs. When one of the input is `None`, it raise an exception. Note that you can
use the concatenation, like any other operator, at a meta-class level, meaning
you can actually concatenate data model types.

### Concatenation Table

| `x1`   | `x2`   | Concat (`+`)      |
| ------ | ------ | ----------------- |
| `x1`   | `x2`   | `x1 + x2`         |
| `x1`   | `None` | `Exception`       |
| `None` | `x2`   | `Exception`       |
| `None` | `None` | `Exception`       |

### Concatenation Example

```python

async def main():
    inputs = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    outputs = x1 + x2

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="concatenation",
        description="Illustrate the use of concatenate",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

![concatenation](../../assets/concatenation.png)

### Logical And

The `logical_and` is a concatenation that instead of raising an `Exception`,
output a `None`. This operator should be used, when you have to concatenate
a data model with an another one that can be `None`, like a `Branch` output.

### Logical And Table

| `x1`   | `x2`   | Logical And (`&`) |
| ------ | ------ | ----------------- |
| `x1`   | `x2`   | `x1 + x2`         |
| `x1`   | `None` | `None`            |
| `None` | `x2`   | `None`            |
| `None` | `None` | `None`            |

### Logical And Example

```python
class Critique(synalinks.DataModel):
    critique: str = synalinks.Field(
        description="The critique of the answer",
    )

async def main():
    inputs = synalinks.Input(data_model=Query)
    (x1, x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        return_decision=False,
    )(inputs)
    x3 = x0 & x1
    x4 = x0 & x2
    x5 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(x3)
    x6 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(x4)
    x7 = x5 | x6
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(x7)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="logical_and",
        description="Illustrate the use of logical and",
    )
```

![logical_and](../../assets/logical_and.png)

### Logical Or

The `logical_or` is used when you want to combine two data models, but you can
accomodate that one of them is `None`. Another use, is to gather the outputs of
a `Branch`, as only one branch is active, it allows you merge the branches outputs 
into a unique data model.

### Logical Or Table

| `x1`   | `x2`   | Logical Or (`|`) |
| ------ | ------ | ---------------- |
| `x1`   | `x2`   | `x1 + x2`        |
| `x1`   | `None` | `x1`             |
| `None` | `x2`   | `x2`             |
| `None` | `None` | `None`           |

### Logical Or Example

```python
async def main():
    inputs = synalinks.Input(data_model=Query)
    (x1, x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        return_decision=False,
    )(x0)
    outputs = x1 | x2

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="logical_or",
        description="Illustrate the use of logical or",
    )
```

![logical_or](../../assets/logical_or.png)


## Conclusion

In this tutorial, we explored the fundamental concepts of controlling information
flow within Synalinks programs. We introduced the creation of parallel branches, 
decision-making processes, and conditional branching, all of which are essential 
for building dynamic and robust applications.

### Key Takeaways

- **Parallel Branches**: We demonstrated how to run modules in parallel using 
    the same inputs, leveraging asyncio for concurrent execution. 
    This approach enhances performance and allows for simultaneous processing of tasks.

- **Decision-Making**: We introduced decision-making as a form of single-label 
    classification, enabling the system to classify inputs based on predefined 
    questions and labels. This ensures that the system's responses are structured 
    and adhere to the specified schemas.

- **Conditional Branching**: We explored the use of the Branch module to route 
    input data models based on decisions, allowing for conditional execution of 
    branches. This feature is essential for creating adaptive and context-aware 
    applications.

- **Data Model Operators**: We discussed various data model operators, such as 
    concatenation, `logical_and`, and `logical_or`. These operators enable 
    sophisticated data manipulation and flow control, ensuring robust program
    execution even when branches output `None`.