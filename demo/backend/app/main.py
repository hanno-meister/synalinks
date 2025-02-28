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

# Load your program
program = synalinks.Program.load("checkpoint.program.json")


@app.post("/v1/chat_completion")
async def chat_completion(messages: synalinks.ChatMessages):
    result = await program(messages)
    return result.json() if result else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
