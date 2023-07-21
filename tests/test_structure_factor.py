import unittest
import matplotlib.pyplot as plt
from structure_factor_utils.structure_factor import ElasticNeutronFactor, StructureFactorArtist

class TestStructureFactorLaMnO3(unittest.TestCase):
    def test_from_cif(self):
        enfactor = ElasticNeutronFactor.from_cif("tests/0.1_LaMnO3.mcif")
        test_scattering_vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.32, 12392, 31934]
        ]
        structure_factors = []
        for scattering_vector in test_scattering_vectors:
            sf = enfactor.get_structure_factor(scattering_vector)
            structure_factors.append(sf)
            print(f"Q={scattering_vector}: {sf}")

        self.assertEqual(structure_factors, [
            ((22.237407+0j), 0j, 0j),
            ((22.237407+0j), 0j, 0j),
            ((-22.237407+2.24740819378772e-14j), 0j, 0j),
            ((22.237407-7.943598800032408e-10j), 0j, 0j)
        ])

    def test_artist(self):
        enfactor = ElasticNeutronFactor.from_cif("tests/0.1_LaMnO3.mcif")
        ax = plt.gca()
        artist = StructureFactorArtist(ax)
        artist.draw_structure_factor(enfactor, [0, 1, 0], [0, 0, 1], (100, 100))

if __name__ == '__main__':
    unittest.main()