import os
import synalinks
import asyncio
import psutil
from dotenv import load_dotenv
from fastapi import FastAPI

import litellm
litellm._turn_on_debug()

# Uncomment for streaming apps
# from fastapi.responses import StreamingResponse

# Import your input data model for your API
from my_project.src.data_models import Query
# Import your custom module
from my_project.src.modules import AnswerWithChainOfThought

# Load the .env variables
load_dotenv()

# Clear Synalinks context
synalinks.clear_session()

async def create_program():
    language_model = synalinks.LanguageModel(
        model=os.environ.get(
            "LANGUAGE_MODEL",
            "ollama/deepseek-r1",
        ),
        api_base=os.environ.get("MODEL_API_BASE", None),
    )        
    inputs = synalinks.Input(data_model=Query)
    outputs = await AnswerWithChainOfThought(
        language_model=language_model,
    )(inputs)

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="my-project",
        description="My awesome project",
    )

program = asyncio.run(create_program())

# Load your serialized application
program_variables_filepath = "checkpoint.program.variables.json"
if os.path.exists(program_variables_filepath):
    program.load_variables(program_variables_filepath)

# Setup FastAPI
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/system_check")
def system_check():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage("/")
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_info.percent,
        "disk_usage": disk_usage.percent
    }

@app.post("/v1/my_project")
async def my_project(inputs: Query):
    result = await program(inputs)
    # Uncomment for streaming apps
    # return StreamingResponse(result, media_type="application/json") if result else None
    return result.get_json() if result else None