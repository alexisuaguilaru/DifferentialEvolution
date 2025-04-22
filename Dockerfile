FROM python:3.11-slim
WORKDIR /DIFF_EVOL

# Install requirements
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy necessary code and function
COPY ./DifferentialEvolution ./DifferentialEvolution
COPY ./TestingFunctions ./TestingFunctions
COPY ScriptExperiments.py .

# Execute experiment
CMD ["python","ScriptExperiments.py"]