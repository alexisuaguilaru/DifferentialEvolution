FROM python:3.12.5-alpine
WORKDIR /DIFF_EVOL

# Install requirements
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r 
RUN pip install --ignore-requires-python opfunu==1.0.4

# Copy necessary code and function
COPY ./DifferentialEvolution ./DifferentialEvolution
COPY ./TestingFunctions ./TestingFunctions
COPY ScriptExperiments.py .

# Execute experiment
CMD ["python","ScriptExperiments.py"]