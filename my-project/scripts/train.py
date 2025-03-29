#!/usr/bin/python3
import os
import synalinks
import asyncio
import json
import numpy as np
from my_project.src.data_models import Query
from my_project.src.data_models import Answer
from my_project.src.modules import AnswerWithChainOfThought

# Clear Synalinks context
synalinks.clear_session()

# The training is done in batch, the batch size specify the 
# number of parralel program execution performed to train 
# the application. The reward is averaged per batch yielding 
# a better estimation of your program success.
BATCH_SIZE=32

# The epochs refer to the number of time the whole dataset is
# proccessed. At the end of each epochs, the optimization is
# performed. So the epochs is the number of successive optimization.
EPOCHS=4

def load_dataset(
    input_data_model,
    output_data_model,
    x_train_filepath="datasets/x_train.json",
    y_train_filepath="datasets/y_train.json",
    x_test_filepath="datasets/x_test.json",
    y_test_filepath="datasets/y_test.json",
):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    with open(x_train_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            x_train.append(input_data_model(**data_point))
    
    with open(y_train_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            y_train.append(output_data_model(**data_point))
        
    with open(x_test_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            x_test.append(input_data_model(**data_point))
            
    with open(y_test_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            y_test.append(output_data_model(**data_point))
    
    # Convert the dataset into numpy arrays
    
    x_train = np.array(x_train, dtype="object")
    y_train = np.array(y_train, dtype="object")
    
    x_test = np.array(x_test, dtype="object")
    y_test = np.array(y_test, dtype="object")

    return (x_train, y_train), (x_test, y_test)

async def train_program():
    language_model = synalinks.LanguageModel(
        model=os.environ.get(
            "LANGUAGE_MODEL",
            "ollama/deepseek-r1"
        ),
    )
    
    embedding_model = synalinks.EmbeddingModel(
        model=os.environ.get(
            "EMBEDDING_MODEL",
            "ollama/all-minilm"
        ),
    )
    
    inputs = synalinks.Input(data_model=Query)
    outputs = await AnswerWithChainOfThought(
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="my-project",
        description="My awesome project",
    )
    
    embedding_model = synalinks.EmbeddingModel(
        model=os.environ.get(
            "EMBEDDING_MODEL",
            "ollama/all-minilm",
        ),
    )
    
    synalinks.utils.plot_program(
        program,
        show_module_names=True,
        show_schemas=True,
    )
    
    checkpoint_filepath = "checkpoint.program.variables.json"
    
    program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_reward",
        save_variables_only=True,
        mode="max",
        save_best_only=True,
    )
    
    program.compile(
        reward=synalinks.rewards.CosineSimilarity(
            # Filter to keep only the `answer` field in order to compute the reward
            in_mask=["answer"],
            # The embedding model to use to compute the similarity
            embedding_model=embedding_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[
            synalinks.metrics.F1Score(in_mask=["answer"]),
        ],
    )
    
    (x_train, y_train), (x_test, y_test) = load_dataset(Query, Answer)
    
    history = await program.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[program_checkpoint_callback],
    )
    
    synalinks.utils.plot_history(
        history,
    )
    
def main():
    asyncio.run(train_program())

if __name__ == "__main__":
    main()