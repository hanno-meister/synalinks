import marimo

__generated_with = "0.11.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import synalinks

    synalinks.backend.clear_session()
    return mo, synalinks


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Your first programs

        The main concept of Synalinks, is that an application (we call it a `Program`)
        is a computation graph with JSON data (called `JsonDataModel`) as edges and
        `Operation`s as nodes. What set apart Synalinks from other similar frameworks
        like DSPy or AdalFlow is that we focus on graph-based systems but also that
        it allow users to declare the computation graph using a Functional API inherited
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
        """
    )
    return


@app.cell
def _(synalinks):
    # Now we can define the data models that we are going to use in the notebook.
    # Note that Synalinks use Pydantic as default data backend, which is compatible with FastAPI and structured output.

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

    return AnswerWithThinking, Query


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Functional API

        You can program your application using 3 different ways, let's start with the Functional way.

        In this case, you start from `Input` and you chain modules calls to specify the programs's structure, and finally, you create your program from inputs and outputs:
        """
    )
    return


@app.cell
async def _(AnswerWithThinking, Query, synalinks):
    language_model = synalinks.LanguageModel(
        model="ollama_chat/deepseek-r1",
    )

    _x0 = synalinks.Input(data_model=Query)
    _x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(_x0)

    program = synalinks.Program(
        inputs=_x0,
        outputs=_x1,
        name="chain_of_thought",
        description="Usefull to answer in a step by step manner.",
    )
    return language_model, program


@app.cell
def _(program):
    # You can print a summary of your program in a table format
    # which is really usefull to have a quick overview of your application

    program.summary()
    return


@app.cell
def _(mo, program, synalinks):
    # Or plot your program in a graph format

    synalinks.utils.plot_program(
        program,
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Subclassing the `Program` class

        Now let's try to program it using another method, subclassing the `Program` class.

        In that case, you should define your modules in `__init__()` and you should implement the program's structure in `call()`.
        """
    )
    return


@app.cell
def _(AnswerWithThinking, language_model, synalinks):
    class ChainOfThought(synalinks.Program):
        """Usefull to answer in a step by step manner.

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

    program_1 = ChainOfThought(language_model=language_model)
    return ChainOfThought, program_1


@app.cell
def _(program_1):
    program_1.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that the program isn't actually built, this behavior is intended its means that it can accept any king of input, making the program truly generalizable. Now we can explore the last way of programming as well as illustrate one of the key feature of Synalinks, composability.

        ## Using `Sequential` program

        In addition to the other ways of programming, `Sequential` is a special case of programs where the program is purely a stack of single-input, single-output modules.

        In this example, we are going to re-use the `ChainOfThought` program that we defined previously, illustrating the modularity of the framework.
        """
    )
    return


@app.cell
def _(ChainOfThought, Query, language_model, synalinks):
    program_2 = synalinks.Sequential(
        [
            synalinks.Input(data_model=Query),
            ChainOfThought(language_model=language_model),
        ],
        name="chain_of_thought",
        description="Usefull to answer in a step by step manner.",
    )
    program_2.summary()
    return (program_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Running your programs""")
    return


@app.cell
async def _(Query, program_2):
    result = await program_2(Query(query="What are the key aspects of human cognition?"))
    print(result.pretty_json())
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that we know how to program applications, we can learn how to control
        the data flow in the next lesson: [Control Flow](#)
        """
    )
    return


if __name__ == "__main__":
    app.run()
