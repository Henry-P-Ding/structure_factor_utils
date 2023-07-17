import unittest
import matplotlib.pyplot as plt
from structure_factor_utils.structure_factor import ElasticNeutronFactor, StructureFactorArtist

class TestStructureFactorLaMnO3(unittest.TestCase):
    def test_from_cif(self):
        enfactor = ElasticNeutronFactor.from_cif("tests/0.1_LaMnO3.mcif")

    def test_artist(self):
        enfactor = ElasticNeutronFactor.from_cif("tests/0.1_LaMnO3.mcif")
        fig, ax = plt.gcf(), plt.gca()
        artist = StructureFactorArtist(ax)
        artist.draw_structure_factor(enfactor, [1, 0, 0], [0, 1, 0], (100, 100))
        fig.savefig("tests/test_output/test_artist.png")

if __name__ == '__main__':
    unittest.main()