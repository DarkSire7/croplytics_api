# 1. Start with a lightweight Python base image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy ONLY the requirements file first
COPY requirements.txt .

# 4. Install your Python libraries
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL your other files into the container
# (main.py, all your .joblib files, etc.)
COPY . .

# 6. Expose the port your app will run on
EXPOSE 8000

# 7. The command to start your FastAPI server
# We use "0.0.0.0" to allow traffic from outside the container
CMD uvicorn main:app --host 0.0.0.0 --port $PORT