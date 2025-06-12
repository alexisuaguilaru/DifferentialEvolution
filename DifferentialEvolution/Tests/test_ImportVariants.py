import pytest

from DifferentialEvolution import DifferentialEvolutionVariant

@pytest.mark.parametrize(('VariantImport','VariantName'),[
    ('DifferentialEvolution','Base'),
    ('DifferentialEvolution_FixedRandomSample','FixedRandomSample'),
    ('DifferentialEvolution_ProportionalRandomSample','ProportionalRandomSample'),
    ('DifferentialEvolution_Agglomerative','Agglomerative'),
])
def test_ImportVariants(VariantImport,VariantName):
    assert getattr(__import__('DifferentialEvolution.Variants',fromlist=[VariantImport]),VariantImport) == DifferentialEvolutionVariant(VariantName)
