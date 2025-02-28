import marimo

__generated_with = "0.11.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import synalinks

    return mo, synalinks


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Training Programs

        Like in machine learning, a LM application needs to be trained. In that case, we
        don't update the weights of the model, but optimize the prompts by automatically
        picking the best examples or generate hints in order to help the program to 
        perform better on your dataset.

        For this lesson we are going to work on GSM8k a well known dataset of grade school
        math word problems. Nowedays, most (all?) public datasets have been leaked, meaning
        that their test set have been included in the LM trainset. This basically means
        that the baseline score won't give you much information about the reasoning abilities
        of the underlying language model (but more about its capability to remember),
        however it is still interesing to have it as a baseline to evaluate the progress 
        of the programs training and the neuro-symbolic methods used or if you use small
        models like here.

        First, let's have a look at the dataset.
        """
    )
    return


@app.cell
def _(synalinks):
    gsm8k_input_data_model = synalinks.datasets.gsm8k.get_input_data_model()
    print("GSM8K input schema:\n")
    print(gsm8k_input_data_model.pretty_schema())
    return (gsm8k_input_data_model,)


@app.cell
def _(synalinks):
    gsm8k_output_data_model = synalinks.datasets.gsm8k.get_output_data_model()
    print("GSM8K output schema:\n")
    print(gsm8k_output_data_model.pretty_schema())
    return (gsm8k_output_data_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Programming the pipeline

        Now let's make a simple baseline program like in the first lessons
        For this example we are going to use the data model from GSM8k.
        """
    )
    return


@app.cell
async def _(gsm8k_input_data_model, gsm8k_output_data_model, synalinks):
    language_model = synalinks.LanguageModel(
        model="ollama_chat/deepseek-r1",
    )

    _x0 = synalinks.Input(data_model=gsm8k_input_data_model)
    _x1 = await synalinks.Generator(
        data_model=gsm8k_output_data_model,
        language_model=language_model,
    )(_x0)

    program = synalinks.Program(
        inputs=_x0,
        outputs=_x1,
        name="chain_of_thought",
        description="Usefull to answer in a step by step manner.",
    )
    return language_model, program


@app.cell(hide_code=True)
def _(mo):
    load_data = mo.ui.run_button(label="Load dataset")
    load_data.center()
    return (load_data,)


@app.cell
def _(load_data, mo, synalinks):
    mo.stop(not load_data.value)
    # Now we can load the dataset
    with mo.status.spinner(title="Loading dataset...") as _spinner:
        (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
        _spinner.update("Done.")
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Compiling the program

        For this example, we are going to select the `RandomFewShot` optimizer.
        The reward fucntion will be `ExactMatch` masked to match only the numerical answer.
        While the additional metric will be the `F1Score` masked to process only the LMs thinking.

        This metric will give us an indication to see if the chain of thought match with the dataset one.
        """
    )
    return


@app.cell
def _(program, synalinks):
    program.compile(
        optimizer=synalinks.optimizers.RandomFewShot(),
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        metrics=[
            synalinks.metrics.F1Score(in_mask=["thinking"]),
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Training

        ### What do "sample", "batch", and "epoch" mean?

        - **Sample**: A sample is one element of a dataset. For example, one DataModel
            is one sample.
        - **Batch**: A batch is a set of N samples. The samples in a batch are processed
            independently, in parallel. During training, a batch result in only one
            program update. A batch approximates the input distribution better than a
            single input. The larger the batch, the better the approximation; however a
            larger batch will take longer to process and still result in only one update.
        - **Epochs**: A epochs is an arbitrarly cutoff, generally defined as "one pass
            over the entire dataset", used to separate training into distinct phases,
            which is usefull for logging and periodic evaluation. When using 
            `validation_split` or `validation_data` with the `fit` method of Synalinks
            programs, evaluation will be run at the end of every epoch.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, x_test, x_train):
    epochs = mo.ui.slider(start=1, stop=64, value=10, label="Epochs")
    batch_size = mo.ui.slider(start=1, stop=64, value=32, label="Batch size")
    train_samples = mo.ui.slider(
        start=1, stop=len(x_train), value=50, label="Train Samples"
    )
    test_samples = mo.ui.slider(start=1, stop=len(x_test), value=50, label="Test Samples")
    return batch_size, epochs, test_samples, train_samples


@app.cell(hide_code=True)
def _(epochs, mo):
    mo.hstack([epochs, mo.md(f"Epochs: {epochs.value}")])
    return


@app.cell(hide_code=True)
def _(batch_size, mo):
    mo.hstack([batch_size, mo.md(f"Batch size: {batch_size.value}")])
    return


@app.cell(hide_code=True)
def _(mo, train_samples):
    mo.hstack([train_samples, mo.md(f"Nb train samples: {train_samples.value}")])
    return


@app.cell(hide_code=True)
def _(mo, test_samples):
    mo.hstack([test_samples, mo.md(f"Nb test samples: {test_samples.value}")])
    return


@app.cell(hide_code=True)
def _(mo):
    start_training = mo.ui.run_button(label="Start training")
    start_training.center()
    return (start_training,)


@app.cell
async def _(
    batch_size,
    epochs,
    mo,
    program,
    start_training,
    synalinks,
    test_samples,
    train_samples,
    x_test,
    x_train,
    y_test,
    y_train,
):
    mo.stop(not start_training.value)
    # Where to save the best performing program
    checkpoint_filepath = "checkpoint.program.json"

    _program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_reward",
        mode="max",
        save_best_only=True,
    )

    # For the purpose of the tutorial, we'll only train on the first N samples

    history = await program.fit(
        epochs=epochs.value,
        batch_size=batch_size.value,
        x=x_train[: train_samples.value],
        y=y_train[: train_samples.value],
        validation_data=(x_test[: test_samples.value], y_test[: test_samples.value]),
        callbacks=[_program_checkpoint_callback],
    )
    return checkpoint_filepath, history


@app.cell
def _(history, synalinks):
    synalinks.utils.plot_history(history)
    return


if __name__ == "__main__":
    app.run()
