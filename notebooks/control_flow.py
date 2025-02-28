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
        # Control Flow

        Controlling the flow of information in a program is an essential feature of any LM framework.
        In Synalinks, we implemented it in circuit-like fashion, where the flow of information can be 
        conditionaly or logically restricted to only flow in a subset of a computation graph.

        ## Parallel Branches

        To create parallel branches, all you need to do is using the same inputs when declaring the modules.
        Then Synalinks will automatically detect them and run them in parrallel with asyncio.
        """
    )
    return


@app.cell
async def _(synalinks):
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

    language_model = synalinks.LanguageModel(model="ollama_chat/deepseek-r1")
    _x0 = synalinks.Input(data_model=Query)
    _x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(_x0)
    _x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(_x0)

    program = synalinks.Program(
        inputs=_x0,
        outputs=[_x1, _x2],
        name="parallel_branches",
        description="Illustrate the use of parallel branching",
    )
    return AnswerWithThinking, Query, language_model, program, synalinks


@app.cell
def _(mo, program, synalinks):
    synalinks.utils.plot_program(
        program,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Decisions

        Decisions in Synalinks can be viewed as a single label classification, they allow
        the system to classify the inputs based on a question and labels to choose from.
        The labels are used to create on the fly a Enum schema that ensure, thanks to
        constrained structured output, that the system will answer one of the provided labels.
        """
    )
    return


@app.cell
async def _(Query, language_model, synalinks):
    _x0 = synalinks.Input(data_model=Query)
    _x1 = await synalinks.Decision(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        language_model=language_model,
    )(_x0)

    program_1 = synalinks.Program(
        inputs=_x0,
        outputs=_x1,
        name="decision_making",
        description="Illustrate the decision making process",
    )
    return (program_1,)


@app.cell
def _(mo, program_1, synalinks):
    synalinks.utils.plot_program(
        program_1,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Conditional Branches

        To make conditional branches, we will need the help of a core module: The Branch
        module. This module use a decision and route the input data model to the selected
        branch. When a branch is not selected, that branch output a None.
        """
    )
    return


@app.cell
async def _(AnswerWithThinking, Query, language_model, synalinks):
    class Answer(synalinks.DataModel):
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    _x0 = synalinks.Input(data_model=Query)
    (_x1, _x2) = await synalinks.Branch(
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
    )(_x0)

    program_2 = synalinks.Program(
        inputs=_x0,
        outputs=[_x1, _x2],
        name="conditional_branches",
        description="Illustrate the conditional branches",
    )
    return Answer, program_2


@app.cell
def _(mo, program_2, synalinks):
    synalinks.utils.plot_program(
        program_2,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data Models Operators

        Synalinks implement few operators that works with data models, some of them are
        straightforward, like the concatenation, implemented in the Python `+` operator. 
        But others like the `logical_and` and `logical_or` implemented respectively 
        in the `&` and `|` operator are more difficult to grasp at first. As explained
        above, in the conditional branches, the branch not selected will have a None 
        as output. To account that fact and to implement logical flows, we need operators
        that can work with them.

        ### Concatenation

        The concatenation, consist in creating a data model that have the fields of both
        inputs. When one of the input is `None`, it raise an exception. Note that you can
        use the concatenation, like any other operator, at a meta-class level, meaning
        you can actually concatenate data model types.

        Table:

        | `x1`   | `x2`   | Concat (`+`)      |
        | ------ | ------ | ----------------- |
        | `x1`   | `x2`   | `x1 + x2`         |
        | `x1`   | `None` | `Exception`       |
        | `None` | `x2`   | `Exception`       |
        | `None` | `None` | `Exception`       |
        """
    )
    return


@app.cell
async def _(AnswerWithThinking, Query, language_model, synalinks):
    _x0 = synalinks.Input(data_model=Query)
    _x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(_x0)
    _x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(_x0)
    _x3 = _x1 + _x2

    program_3 = synalinks.Program(
        inputs=_x0,
        outputs=_x3,
        name="concatenation",
        description="Illustrate the use of concatenate",
    )
    return (program_3,)


@app.cell
def _(mo, program_3, synalinks):
    synalinks.utils.plot_program(
        program_3,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Logical And

        The `logical_and` is a concatenation that instead of raising an `Exception`,
        output a `None`. This operator should be used, when you have to concatenate
        a data model with an another one that can be `None`, like a `Branch` output.


        Table:

        | `x1`   | `x2`   | Logical And (`&`) |
        | ------ | ------ | ----------------- |
        | `x1`   | `x2`   | `x1 + x2`         |
        | `x1`   | `None` | `None`            |
        | `None` | `x2`   | `None`            |
        | `None` | `None` | `None`            |
        """
    )
    return


@app.cell
async def _(Answer, AnswerWithThinking, Query, language_model, synalinks):
    class Critique(synalinks.DataModel):
        critique: str = synalinks.Field(
            description="The critique of the answer",
        )

    _x0 = synalinks.Input(data_model=Query)
    (_x1, _x2) = await synalinks.Branch(
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
    )(_x0)
    _x3 = _x0 & _x1
    _x4 = _x0 & _x2
    _x5 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(_x3)
    _x6 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(_x4)
    _x7 = _x5 | _x6
    _x8 = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(_x7)

    program_4 = synalinks.Program(
        inputs=_x0,
        outputs=_x8,
        name="logical_and",
        description="Illustrate the use of logical and",
    )
    return Critique, program_4


@app.cell
def _(mo, program_4, synalinks):
    synalinks.utils.plot_program(
        program_4,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Logical Or

        The `logical_or` is used when you want to combine two data models, but you can
        accomodate that one of them is `None`. Another use, is to gather the outputs of
        a `Branch`, as only one branch is active, it allows you merge the branches outputs 
        into a unique data model.


        Table:

        | `x1`   | `x2`   | Logical Or (`|`) |
        | ------ | ------ | ---------------- |
        | `x1`   | `x2`   | `x1 + x2`        |
        | `x1`   | `None` | `x1`             |
        | `None` | `x2`   | `x2`             |
        | `None` | `None` | `None`           |
        """
    )
    return


@app.cell
async def _(Answer, AnswerWithThinking, Query, language_model, synalinks):
    _x0 = synalinks.Input(data_model=Query)
    (_x1, _x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking, language_model=language_model
            ),
        ],
        return_decision=False,
    )(_x0)
    _x3 = _x1 | _x2

    program_5 = synalinks.Program(
        inputs=_x0,
        outputs=_x3,
        name="logical_or",
        description="Illustrate the use of logical or",
    )
    return (program_5,)


@app.cell
def _(mo, program_5, synalinks):
    synalinks.utils.plot_program(
        program_5,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )
    return


@app.cell(hide_code=True)
async def _(mo):
    mo.md(
        r"""
        The next step is now to understand the basic concepts to train/optimize 
        Synalinks programs: [Rewards, Metrics & Optimizers](#)
        """
    )
    return


if __name__ == "__main__":
    app.run()
