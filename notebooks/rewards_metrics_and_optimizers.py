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
        # Rewards, Metrics & Optimizers

        ## Understanding Rewards

        `Reward`s are an essential part of reinforcement learning frameworks. 
        They are typically float values (usually between 0.0 and 1.0, but they can be 
        negative also) that guide the process into making more efficient decisions or 
        predictions. During training, the goal is to maximize the reward function. 
        The reward gives the system an indication of how well it performed for that task.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        r"""
        graph LR
        A[Training Data] -->|Provide x:DataModel| B[Program];
        B -->|Generate y_pred:JsonDataModel| C[Reward];
        A -->|Provide y_true:DataModel| C;
        C -->|Compute reward:Float| D[Optimizer];
        D -->|Update trainable_variable:Variable| B;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This reinforcement loop is what makes possible for the system to learn by
        repeatedly making predictions and refining its knowledge/methodology in order 
        to maximize the reward.

        All rewards consist of a function or program that takes two inputs:

        - `y_pred`: The prediction of the program.
        - `y_true`: The ground truth/target value provided by the training data.

        In Synalinks, we provide for several built-in rewards but it is also possible to
        easily create new rewards if you needs to. Overall the choice will depend on the
        task to perform. You can have a look at the rewards provided in the [API section](#).

        ### Understanding Metrics

        `Metric`s are scalar values that are monitored during training and evaluation.
        These values are used to know which program is best, in order to save it. Or to 
        provide additional information to compare different architectures with each others.
        Unlike `Reward`s, a `Metric` has a state (a `Variable`) that is updated over time
        and is not used during training, meaning the metric value is not backpropagated. 
        Additionaly every reward function can be used as metric. You can have a look at the 
        metrics provided in the [API section](#).

        ### Filtering y

        Sometimes, your program have to output a complex JSON but you want to evaluate
        just part of it. This could be because your training data only include a subset
        of the JSON, or because the additonal fields were added only to help the LMs.
        In that case, you have to filter out or filter in your predictions and ground
        truth. Meaning that you want to remove or keep respectively only specific fields
        of your JSON data. This can be achieved by adding a `out_mask` or `in_mask` list
        parameter containing the keys to remove or keep for evaluation. This parameters
        can be added to both reward and metrics. Like in the above example where we only
        keep the field `answer` to compute the rewards and metrics.

        ### Understanding Optimizers

        Optimizers are systems that handle the update of the module's state in order to
        make them more performant. They are in charge of backpropagating the rewards 
        from the training process and select or generate examples and hints for the LMs.

        Here is an example of program compilation, which is how you configure the reward,
        metrics, and optimizer:
        """
    )
    return


@app.cell
async def _(synalinks):
    synalinks.backend.clear_session()

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

    program.compile(
        reward=synalinks.rewards.CosineSimilarity(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[
            synalinks.metrics.F1Score(in_mask=["answer"]),
        ],
    )
    return AnswerWithThinking, Query, language_model, program


@app.cell(hide_code=True)
async def _(mo):
    mo.md(
        r"""
        Now that you understand the basic concepts to train/optimize Synalinks programs,
        we can actually train one: [Training Programs](#)
        """
    )
    return


if __name__ == "__main__":
    app.run()
