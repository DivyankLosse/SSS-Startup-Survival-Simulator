FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
