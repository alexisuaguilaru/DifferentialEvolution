import pytest

from DifferentialEvolution import DifferentialEvolutionVariant

@pytest.mark.parametrize(('VariantImport','VariantName'),[
    ('DifferentialEvolution','Base'),
    ('DifferentialEvolution_Agglomerative','Agglomerative'),
    ('DifferentialEvolution_FixedRandomSample','FixedRandomSample'),
    ('DifferentialEvolution_ProportionalRandomSample','ProportionalRandomSample'),
])
def test_ImportVariants(VariantImport,VariantName):
    assert getattr(__import__('DifferentialEvolution.Variants',fromlist=[VariantImport]),VariantImport) == DifferentialEvolutionVariant(VariantName)
