FROM python:3.13

COPY ./requirements.txt ./requirements.txt

RUN apt update && apt install -y graphviz

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . ./

CMD ["fastapi", "run", "my_project/src/main.py", "--host", "0.0.0.0", "--port", "8000"]