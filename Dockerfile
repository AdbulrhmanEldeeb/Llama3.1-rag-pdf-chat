# Use a slim version of Python 3.12 as the base image
FROM python:3.12-slim 

# Set the working directory to /app
WORKDIR /code

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose port 8080 for the application
EXPOSE 8080

# Specify the command to run the application
ENTRYPOINT ["streamlit", "run", "app.py"]

# Set default command-line arguments for the ENTRYPOINT
CMD ["--server.port", "8080"]
