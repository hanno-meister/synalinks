# Building a REST API

The optimal approach to developing web-apps or micro-services using Synalinks involves building REST APIs and deploying them. You can deploy these APIs locally to test your system or on a cloud provider of your choice to scale to millions of users.

For this purpose, you will need to use FastAPI, a Python library that makes it easy and straightforward to create REST APIs. If you use the default backend, the DataModel will be compatible with FastAPI as their both use Pydantic.

In this tutorial we are going to make a backend that run locally to test our system.

## Project structure

Your project structure should look like this:

```shell
demo/
├── backend/
│   ├── app/
│   │   ├── checkpoint.program.json
│   │   └── main.py
│   ├── requirements.txt
│   ├── Dockerfile
├── frontend/
│   └── ... (your frontend code)
├── scripts/
│   └── train.py (refer to the code examples to learn how to train programs)
├── docker-compose.yml
├── .env.backend
└── README.md
```

## Your `requirements.txt` file

Import additionally any necessary dependency

```txt title="requirements.txt"
fastapi[standard]
uvicorn
synalinks
openinference-instrumentation-litellm
arize-otel
```

## Creating your endpoint using FastAPI and SynaLinks

Now you can create you endpoint using FastAPI.

```python title="main.py"
import argparse
import logging
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

import synalinks

# Import open-telemetry dependencies
from arize.otel import register
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Load the environment variables
load_dotenv()

# Setup OTel via Arize Phoenix convenience function
tracer_provider = register(
    space_id = os.environ["ARIZE_SPACE_ID"], # in app space settings page
    api_key = os.environ["ARIZE_API_KEY"], # in app space settings page
    project_name = os.environ["ARIZE_PROJECT_NAME"], # name this to whatever you would like
)

LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Set up FastAPI
app = FastAPI()

# The dictionary mapping the name of your custom modules to their class
custom_modules = {}

# Load your program
program = synalinks.Program.load(
    "checkpoint.program.json",
    custom_modules=custom_modules,
)

@app.post("/v1/chat_completion")
async def chat_completion(messages: synalinks.ChatMessages):
    logger.info(messages.pretty_json())
    try:
        result = await program(messages)
        if result:
            logger.info(result.pretty_json())
            return result.get_json()
        else:
            return None
    except Exception as e:
        logger.error(f"Error occured: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
```

## Creating the Dockerfile

Here is the dockerfile to use according to FastAPI documentation.

```Dockerfile title="Dockerfile"
FROM python:3.13

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
```

## The docker compose file

And finally your docker compose file.

```yml title="docker-compose.yml"
services:
  arizephoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env.backend
    depends_on:
      - arizephoenix
```

## Launching your backend

Launch your backend using `docker compose`

```shell
cd demo
docker compose up
```

Open you browser to `http://0.0.0.0:8000/docs` and test your API with the FastAPI UI