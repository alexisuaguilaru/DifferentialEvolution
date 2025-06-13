FROM quay.io/jupyter/base-notebook

WORKDIR /DIFF_EVOL
USER root:root

EXPOSE 8888

# Install requirements
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --ignore-requires-python opfunu==1.0.4

# Copy necessary code and function
COPY ./DifferentialEvolution ./DifferentialEvolution
COPY ./TestingFunctions ./TestingFunctions
COPY ProcessResults.py .

# Execute experiment
CMD ["jupyter","lab","--allow-root"]