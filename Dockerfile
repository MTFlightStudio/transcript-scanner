# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Expose port 8080
EXPOSE 8080

# Run the application
CMD streamlit run --server.port 8080 --server.address 0.0.0.0 app.py