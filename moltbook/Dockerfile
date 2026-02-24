FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .
COPY SKILL.md HEARTBEAT.md MESSAGING.md RULES.md package.json ./

CMD ["python3", "-u", "agent.py", "run"]
