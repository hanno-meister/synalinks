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
        # First Steps

        First, install Synalinks, the easiest way is using pip:

        ```shell
        pip install synalinks
        ```

        Or uv (recommended):

        ```shell
        uv pip install synalinks
        ```

        If you want to install it from source (for contributors), then do:

        ```shell
        git clone https://github.com/SynaLinks/Synalinks
        cd Synalinks
        ./shell/uv.sh # Install uv
        ./shell/install.sh # Create the virtual env and install Synalinks
        ```

        After this, open a python file or notebook and check the install:
        """
    )
    return


@app.cell
def _(synalinks):
    print(synalinks.__version__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Synalinks use a global context to ensure that each variable/module
        have a unique name. Clear it at the beginning of your scripts to 
        ensure naming reproductability.
        """
    )
    return


@app.cell
def _(synalinks):
    synalinks.backend.clear_session()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Addtionally, you can install Ollama [here](https://ollama.com/) to run
        Language Models (LMs) locally.

        ## Prompting

        You will notice that there is no traditional prompting involved in 
        Synalinks, everything is described as data models in and out.
        However we use a prompt template, that will tell the system how to 
        construct the prompt automatically.

        The prompt template is a jinja2 template that describe how to render 
        the examples, hints and how to convert them into chat messages:
        """
    )
    return


@app.cell
def _(synalinks):
    print(synalinks.default_prompt_template())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you are making a conversational application, we provide the following template to use.
        To use it, provide this template to the `prompt_template` argument of your `Generator` module.
        Note that this template only works if your module has a `ChatMessages` input.
        """
    )


@app.cell
def _(synalinks):
    print(synalinks.chat_prompt_template())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The template use the XML tags `<system>...</system>`, `<user>...</user>` and
        `<assistant>...</assistant>` to know how to convert the prompt template 
        into messages. You can modify the default template used by using the 
        `prompt_template` argument in Synalinks modules. You can notice also, 
        that we send the inputs's and output's JSON schema to instruct the LMs
        how to answer, you can enable/disable that behavior by using `use_inputs_schema`
        and `use_outputs_schema` in Synalinks modules. Synalinks use constrained
        structured output ensuring that the LMs answer respect the data models
        specification (the JSON schema), and is ready to parse, so in theory
        we don't need it, except if you use it to provide additional information
        to the LMs.

        ## Data Models

        To provide additional information to the LMs, you can use the data models
        `Field`. You can notice that Synalinks use Pydantic as default data backend.
        Allowing Synalinks to be compatible out-of-the-box with structured output 
        and FastAPI.
        """
    )
    return


@app.cell
def _(synalinks):
    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking process",
        )
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    return (AnswerWithThinking,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Usually that will be enough to instruct the LMs, you don't need to modify
        the prompt template. Just by adding additional descriptions to the data
        models fields you can instruct your system to behave as you want. 
        If the system needs general instructions about how to behave, you can 
        use the `hints` argument in Synalinks modules that will be formatted as 
        presented in the prompt template.

        Now you are ready for the next lesson: [First Programs](#)
        """
    )
    return


if __name__ == "__main__":
    app.run()
