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
        # Conversational Applications

        Synalinks is designed to handle conversational applications as well as 
        query-based systems. In the case of a conversational applications, the
        input data model is a list of chat messages, and the output an individual
        chat message. The `Program` is in that case responsible of handling a
        **single conversation turn**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can program our application like you would do with any `Program`. For this example,
        we are going to make a very simple chatbot.

        By default, if no data_model/schema is provided to the `Generator` it will output a `ChatMessage` like output.
        If the data model is `None`, then you can enable streaming.

        **Note:** Streaming is disabled during training and should only be used in the **last** `Generator` of your pipeline.
        """
    )
    return


@app.cell
async def _(synalinks):
    from synalinks.backend import ChatMessage
    from synalinks.backend import ChatRole
    from synalinks.backend import ChatMessages

    language_model = synalinks.LanguageModel(model="ollama_chat/deepseek-r1")

    _x0 = synalinks.Input(data_model=ChatMessages)
    _x1 = await synalinks.Generator(
        language_model=language_model,
        prompt_template=synalinks.chat_prompt_template(),
        streaming=False,  # Marimo chat don't handle streaming yet
    )(_x0)

    program = synalinks.Program(
        inputs=_x0,
        outputs=_x1,
    )

    # Let's plot this program to understand it

    synalinks.utils.plot_program(
        program,
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )
    return ChatMessage, ChatMessages, ChatRole, language_model, program


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running the chatbot inside the notebook

        In this example, we will show you how to run the conversational application inside this reactive notebook.
        """
    )
    return


@app.cell(hide_code=True)
def _(ChatMessage, ChatMessages, ChatRole, mo, program):
    def cleanup_assistant_message(msg):
        start_tok = '<span class="paragraph">'
        end_tok = "</span>"
        msg.content = msg.content[msg.content.find(start_tok) + len(start_tok) :]
        msg.content = msg.content[: msg.content.find(end_tok, 1)]
        return msg

    async def synalinks_program(messages, config):
        chat_history = ChatMessages()
        for msg in messages:
            if msg.role == "user":
                chat_history.messages.append(
                    ChatMessage(
                        role=ChatRole.USER,
                        content=msg.content,
                    )
                )
            else:
                msg = cleanup_assistant_message(msg)
                chat_history.messages.append(
                    ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=msg.content,
                    )
                )
        result = await program(chat_history)
        return result.get("content")

    chat = mo.ui.chat(synalinks_program)
    chat
    return chat, cleanup_assistant_message, synalinks_program


if __name__ == "__main__":
    app.run()
