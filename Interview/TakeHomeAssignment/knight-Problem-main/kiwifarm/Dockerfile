FROM python:3.11

WORKDIR /app

# Install Graphviz
RUN apt-get update && \
    apt-get install -y graphviz && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY knight_problem.py .

CMD ["python", "knight_problem.py"]
