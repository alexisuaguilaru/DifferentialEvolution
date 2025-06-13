# DifferentialEvolution

## Abstract
Repository to experiment with different variants of Differential Evolution based on clustering.

## Affiliation
This project has academic purposes related to the *Universidad Nacional Autónoma de México* [UNAM](https://www.unam.mx/).

## Acknowledge
Se agradece el apoyo prestado por el programa [PAPIME](https://dgapa.unam.mx/index.php/fortalecimiento-a-la-docencia/papime) para la realización del presente proyecto con clave *PE101325: ANALISIS DEL FUNCIONAMIENTO DE EVOLUCION DIFERENCIAL*.

## Contact
Para más información sobre el uso del proyecto, contactar con [Alexis Aguilar](https://github.com/alexisuaguilaru) al correo alexis.uaguilaru@gmail.com

## Installation & Usage
It is first necessary to clone the project:
```bash
git clone https://github.com/alexisuaguilaru/DifferentialEvolution
```

### Using Docker
Docker was used to create a Jupyter environment that contains everything necessary to use this project along with other libraries to perform the optimization of Differential Evolution hyperparameters.

First, the Docker image has to be built, in which it will be possible to interact with the Differential Evolution variants in a Jupyter environment:
```bash
docker build -t diff_evol .
```

Finally, the built container image is executed:
```bash
docker run -p 8888:8888 --mount type=bind,src=./Experiments,dst=/DIFF_EVOL/Experiments diff_evol
```

It opens as if it were any other run performed with ``jupyter lab``. In order to properly preserve the created Jupyter notebooks, they have to be created in the Experiments folder inside the Jupyter environment and can be found in the folder with the same name in the project directory.

### Using Python Scripts
First it is necessary to install the libraries, preferably in a Python environment, using the following commands:
```bash
pip install -r requirements.txt
pip install --ignore-requires-python opfunu==1.0.4
```

This allows making full use of the project and the Differential Evolution variants. To execute the variants as Python scripts and generate the relevant results, it is suggested to consult the example script [MinimalExample.py](./MinimalExample.py).

The example provided can be executed with the following command:
```bash
python MinimalExample.py
```