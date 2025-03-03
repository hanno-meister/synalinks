# Building a REST API

The optimal approach to developing Web-Apps or Micro-Services using Synalinks involves building REST APIs and deploying them. You can deploy these APIs locally to test your system or on a cloud provider of your choice to scale to millions of users.

For this purpose, you will need to use FastAPI, a Python library that makes it easy and straightforward to create REST APIs. If you use the default backend, the DataModel will be compatible with FastAPI as their both use Pydantic.

First, let's see how a production-ready project is structured.

In this tutorial, we will focus only on the backend of your application.

For the purpose of this tutorial, we will skip authentification. But because is is higly dependent on your business usecase/frontend, we will not handle it here.

### Project Structure

```sh
demo/
│
├── backend/
│   ├── app/
│   │   ├── checkpoint.program.json
│   │   └── main.py
│   ├── requirements.txt
│   ├── Dockerfile
├── frontend/
│   └── ... (your frontend code)
├── scripts/
│   ├── export_program.py
│   └── train.py (refer to the code examples to learn how to train programs)
├── docker-compose.yml
├── .env.backend
├── .end.frontend
└── README.md
```

### Setting up your environment variables

```env title=".env.backend"
LLM_PROVIDER=ollama_chat/deepseek-r1
EMBEDDING_PROVIDER=ollama/mxbai-embed-large

LM_PROVIDER_API_BASE=http://localhost:11434
EMBEDDING_PROVIDER_API_BASE=http://localhost:11434

MLFLOW_URL=http://localhost:5000

PROD_CORS_ORIGIN=http://localhost:3000
```

Feel free to add any API key needed for your LM provider

### Creating your endpoint using FastAPI and SynaLinks

```python title="main.py"
import argparse
import logging
import os

import mlflow
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import synalinks

load_dotenv()

mlflow.litellm.autolog()
# Set this to your MLflow server
mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Set up FastAPI
app = FastAPI()

# Set up CORS ORIGIN to avoid connexion problems with the frontend
origins = [
    "http://localhost",
    os.getenv("PROD_CORS_ORIGIN"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The dictionary mapping the name of your custom modules to the class
custom_modules = {}

# Load your program
program = synalinks.Program.load(
    "checkpoint.program.json",
    custom_modules=custom_modules,
)

@app.post("/v1/chat_completion")
async def chat_completion(messages: synalinks.ChatMessages):
    result = await program(messages)
    return result.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
```

### Creating your training script

For obvious reasons, you will need to have a separate logic to train your application. This script will specify the program architecture, training and evaluation procedure and will end up saving your program into a serializable JSON format.

To ease the migration, we'll also make a small script that export the trained program into our backend folder.

```python title="export_program.py"
import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Copy a serialized program to a specified directory."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the file to be copied.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory.",
    )
    args = parser.parse_args()

    if not args.filepath.endswith(".json"):
        raise ValueError("The filepath must ends with `.json`")

    if not os.path.exists(args.output_dir):
        print(f"[*] Output directory does not exist. Creating: '{args.output_dir}'")
        os.makedirs(args.output_dir)

    filename = os.path.basename(args.filepath)
    destination = os.path.join(args.output_dir, filename)
    print(f"[*] Copying file from '{args.filepath}' to '{destination}'...")
    shutil.copy2(args.filepath, destination)
    print(f"[*] Program exported to '{destination}'")


if __name__ == "__main__":
    main()
```

### Creating the backend's Dockerfile

According to [FastAPI documentation](https://fastapi.tiangolo.com/deployment/docker/#dockerfile) here is the right Dockerfile to use for your backend.

```Dockerfile
FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]
```

### Creating your docker-compose.yml file

```yaml
version: '3'
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    env_file:
      - .env.backend
    volumes:
      - ./data:/code/data
```

### Launching your application

```shell
cd demo
docker compose up
```

### Testing your application backend

Open you browser to `http://127.0.0.1/docs` and test your API