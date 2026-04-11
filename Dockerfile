FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install dependencies including uvicorn for FastAPI
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && pip install --no-cache-dir uvicorn

COPY . /code

CMD ["uvicorn", "scripts.api_server:app", "--host", "0.0.0.0", "--port", "7860"]
# uvicorn scripts.api_server:app --host 0.0.0.0