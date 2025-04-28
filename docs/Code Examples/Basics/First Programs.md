# Your First Programs

The main concept of Synalinks, is that an application (we call it a `Program`)
is a computation graph (a Directed Acyclic Graph to be exact) with JSON data (called `JsonDataModel`) as edges and `Operation`s as nodes.

What set apart Synalinks from other similar frameworks like DSPy or AdalFlow is that we focus on graph-based systems but also that it allow users to declare the computation graph using a Functional API inherited
from [Keras](https://keras.io/).

About modules, similar to layers in deep learning applications, modules are
composable blocks that you can assemble in multiple ways. Providing a modular
and composable architecture to experiment and unlock creativity.

Note that each `Program` is also a `Module`! Allowing you to encapsulate them
as you want.

Many people think that what enabled the Deep Learning revolution was compute
and data, but in reality, frameworks also played a pivotal role as they enabled
researchers and engineers to create complex architectures without having to 
re-implement everything from scatch.

```python
import synalinks
import asyncio
# Now we can define the data models that we are going to use in the tutorial.

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking process",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )

# And the language model to use

language_model = synalinks.LanguageModel(
    model="ollama/mistral",
)
```

## Functional API

You can program your application using 4 different ways, let's start with the
Functional way.

In this case, you start from `Input` and you chain modules calls to specify the
programs's structure, and finally, you create your program from inputs and outputs:

```python

async def main():

    x0 = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(x0)

    program = synalinks.Program(
        inputs=x0,
        outputs=x1,
        name="chain_of_thought",
        description="Useful to answer in a step by step manner.",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Subclassing the `Program` class

Now let's try to program it using another method, subclassing the `Program`
class. It is the more complicated one, and reserved for skilled developers or contributors.

In that case, you should define your modules in `__init__()` and you should
implement the program's structure in `call()` and the serialization methods (`get_config` and `from_config`).

```python
class ChainOfThought(synalinks.Program):
    """Useful to answer in a step by step manner.

    The first line of the docstring is provided as description for the program
    if not provided in the `super().__init__()`. In a similar way the name is
    automatically infered based on the class name if not provided.
    """

    def __init__(self, language_model=None):
        super().__init__()
        self.answer = synalinks.Generator(
            data_model=AnswerWithThinking, language_model=language_model
        )

    async def call(self, inputs, training=False):
        x = await self.answer(inputs)
        return x

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)

program = ChainOfThought(language_model=language_model)
```

Note that the program isn't actually built, this behavior is intended its 
means that it can accept any king of input, making the program truly 
generalizable.

## Mixing the subclassing and the `Functional` API

This way of programming is recommended to encapsulate your application while providing an easy to use setup.
It is the recommended way for most users as it avoid making your program/agents from scratch.
In that case, you should implement only the `__init__()` and `build()` methods.

```python

class ChainOfThought(synalinks.Program):
    """Useful to answer in a step by step manner."""

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.language_model = language_model
    
    async def build(self, inputs):
        outputs = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
        )(inputs)

        # Create your program using the functional API
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )

program = ChainOfThought(
    language_model=language_model,
)
```

Like when using the subclassing method, the program will be built on the fly when called for the first time.

## Using the `Sequential` API

In addition, `Sequential` is a special case of program where the program
is purely a stack of single-input, single-output modules.

```python

async def main():

    program = synalinks.Sequential(
        [
            synalinks.Input(
                data_model=Query,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        name="chain_of_thought",
        description="Useful to answer in a step by step manner.",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Running your programs
        
In order to run your program, you just have to call it with the input data model
as argument.

```python
result = await program(
    Query(query="What are the key aspects of human cognition?"),
)
```

## Conclusion
        
Congratulations! You've successfully explored the fundamental concepts of programming
applications using Synalinks.

Now that we know how to program applications, you can learn how to control
the data flow in the next tutorial.

### Key Takeaways

- **Functional API**: Allows you to chain modules to define the program's structure, 
    providing a clear and intuitive way to build applications.
- **Subclassing**: Offers flexibility and control by defining modules and implementing
    the program's structure from scratch within a class.
- **Mixing the subclassing and the Functional API**: Allows to benefit from the
    compositionality of the subclassing while having the ease of use of the functional way of programming.
- **Sequential Programs**: Simplifies the creation of linear workflows, making it easy
    to stack single-input, single-output modules.
- **Modularity and Composability**: Enables the reuse of components, fostering  
    creativity and efficiency in application development.