import pytest

from DifferentialEvolution import DifferentialEvolutionVariant
from .BasicFunctions import F1 , F2 , F3 , Individuals

@pytest.mark.parametrize(('VariantName','ParametersVariant'),[
    ('Base',(0.5,0.5)),
    ('Base',((0.25,0.75),(0.25,0.75))),
    ('Agglomerative',(0.5,0.5,[0.3,0.6])),
    ('Agglomerative',((0.25,0.75),(0.25,0.75),[0.3,0.6])),
    ('FixedRandomSample',(0.5,0.5,[0.5],5)),
    ('FixedRandomSample',((0.25,0.75),(0.25,0.75),[0.5],5)),
    ('FixedRandomSample',(0.5,0.5,[0.5],0.5)),
    ('FixedRandomSample',((0.25,0.75),(0.25,0.75),[0.5],0.5)),
])
def test_Funcionality(VariantName,ParametersVariant):
    diff_evol_class = DifferentialEvolutionVariant(VariantName)

    functions_divergence = []
    for index , func in enumerate([F1,F2,F3]):
        diff_evol = diff_evol_class(func,Individuals)
        _ , snapshots = diff_evol(1000,50,*ParametersVariant)
        if snapshots[-1] > snapshots[0]:
            functions_divergence.append(str(index))

    assert not functions_divergence , ' '.join(functions_divergence)