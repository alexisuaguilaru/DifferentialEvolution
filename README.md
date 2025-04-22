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
<details open>
<summary><h3>With Docker Run</h3></summary>

The configuration file located in ``Experiments/ConfigExperiments.json`` is the main file that has to be modified to generate the different experiments. It contains the configuration of the parameters related to the Differential Evolution variants as well as the parameters for their execution.

First it will be required to build the image of the container with the following command:
```bash
docker build -t diff_evol .
```
Subsequently, the configuration file will have to be modified to adapt it to the experiment or simulations to be executed. Finally, the container is executed with:
```bash
docker run -d -it --rm -v ./Experiments:/DIFF_EVOL/Experiments diff_evol
```
</details>

<details open>
<summary><h3>With Python </h3></summary>

In order to run the ``.py`` script it is first necessary to install the necessary dependencies:
```bash
pip install -r requirements.txt
pip install --ignore-requires-python opfunu==1.0.4
```
Then you have to configure and modify the parameters contained in ``ScriptResults.py`` to suit the experiment to be run. And finally, the script is executed with:
```bash
python ScriptResults.py
```
</details>