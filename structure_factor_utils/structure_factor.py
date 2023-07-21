"""
Utility for calculating scattering structure factors

Structure factor calculations for various scattering methods are implemented in this file. 
The atomic structures and output files interface with the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (ASE).

    Copyright (C) 2023 Henry Ding

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
    USA
"""
from abc import ABC, abstractmethod
import numpy as np
from ase import io as ase_io


class StructureFactorArtist:
    """Draws plots of the structure factor

    :param _axis: axes the structure factor plots are drawn to
    :type _axis: :class:`matplotlib.axes.Axes`
    """

    def __init__(self, axis):
        """Constructor method"""
        self._axis = axis

    @property
    def axis(self):
        """Gets :attr:`~StructureFactorArtist._axis`"""
        return self._axis

    def draw_structure_factor(
        self, structure_factor, plane_vec1, plane_vec2, sample_shape
    ):
        """Draws a color map of a structure factor.

        Scattering vector inputs are chosen in a plane defined by plane_vec1, plane_vec2.

        :param structure_factor: a StructureFactor object that contains system
        :type structure_factor: StructureFactor
        :param plane_vec1: a vector in the reciprocal lattice basis
        :type plane_vec1: tuple[float, float, float]
        :param plane_vec2: a vector in the reciprocal lattice basis
        :type plane_vec2: tuple[float, float, float]
        :param sample_shape: number of lines to sample along plane_vec1, plane_vec2
        :type sample_shape: int
        :raises ValueError: if the structure_factor argument is not an instance of a
            StructureFactor subclass.
        """
        if not isinstance(structure_factor, StructureFactor):
            raise ValueError("structure_factor must be an instance of StructureFactor")

        plane_points = StructureFactorArtist.__sample_plane_points(
            plane_vec1, plane_vec2, sample_shape
        )
        num_points = sample_shape[0] * sample_shape[1]
        # get samples of the norm S of the structure factor in the plane
        S_samples = [  # pylint: disable=invalid-name
            structure_factor.get_sf_norm(
                [plane_points[i, 0], plane_points[i, 1], plane_points[i, 2]]
            )
            for i in range(num_points)
        ]
        # reshape values for S in the shape of a grid with dimensions determined by sample_shape
        S_grid = np.reshape(S_samples, sample_shape)  # pylint: disable=invalid-name

        # drawing to the matplotlib.Axes
        self._axis.imshow(S_grid, origin="lower", extent=[0, 1, 0, 1])
        self._axis.set_xlabel(r"$c_1$")
        self._axis.set_ylabel(r"$c_2$")
        self._axis.set_title(
            rf"$|\mathbf{{S}}(\mathbf{{Q}})|^2$ for $\mathbf{{Q}} = \
c_1\langle{plane_vec1[0]}, {plane_vec1[1]}, {plane_vec1[2]}\rangle + \
c_2\langle{plane_vec2[0]}, {plane_vec2[1]}, {plane_vec2[2]}\rangle$"
        )

    @staticmethod
    def __sample_plane_points(vec1, vec2, sample_shape):
        """Constructs a uniform grid of points on a plane

        A set of points on a plane defined by the vectors vec1, vec2 are sampled. Each point is a
        linear combination c_1 * vec1 + c_2 * vec2 such that there are sample_shape[0],
        sample_shape[1] values of c_1, c_2 evenly distributed on [0, 1] interval.

        :param vec1: a vector that defines the sample plane
        :type vec1: tuple[float, float, float]
        :param vec2: a vector that defines the sample plane
        :type vec2: tuple[float, float, float]
        :param sample_shape: the number of samples for c_1, c_2
        :type sample_shape: tuple[int, int]
        :return: array with shape (N,2) of points on the plane
        :rtype: np.ndarray
        """
        # set of possible values for c1, c2
        c1_space = np.linspace(0, 1, sample_shape[0])
        c2_space = np.linspace(0, 1, sample_shape[1])
        # generate Nx2 array of (c1, c2) tuples
        v1_grid, v2_grid = np.meshgrid(c1_space, c2_space)
        point_coefficients = np.vstack([v1_grid.ravel(), v2_grid.ravel()]).T
        num_points = sample_shape[0] * sample_shape[1]
        # calculate points based on array of (c1, c2) tuples
        return np.tile(point_coefficients[:, 0], (3, 1)).T * np.tile(
            vec1, (num_points, 1)
        ) + np.tile(point_coefficients[:, 1], (3, 1)).T * np.tile(vec2, (num_points, 1))


class StructureFactor(ABC):
    """Abstract base class for the scattering structure factor of a system.

    Subclasses of :class:`StructureFactor` implement different scattering techniques.
    The convention of expressing reciprocal space vectors in the reciprocal lattice basis,
    determinedined by the :class:`ase.Cell.reciprocal()` (fractional coordinates). This includes
    scattering vectors.

    :param name: the name of the system
    :type name: str
    :param atoms: information about atoms in the system unit cell
    :type atoms: :class:`ase.Atoms`
    """

    def __init__(self, name, atoms):
        """Constructor method"""
        self.name = name
        self.atoms = atoms

    @classmethod
    def from_cif(cls, cif_path):
        """Instantiates :class:`StructureFactor` with data from the ``.cif`` or ``.mcif`` file.

        A finite state machine is used to read file data.

        :param cif_path: path to the .cif (.mcif) file
        :type cif_path: str
        :return: :class:`StructureFactor` instance with file data
        :rtype: :class:`StructureFactor`
        """
        # read information from file into an ase.Atoms object
        atoms = ase_io.read(cif_path, format="cif")

        name = "empty"  # identifier for the StructureFactor
        labels = []  # unique labels for each atom in the structure
        moments = np.zeros(
            shape=(len(atoms.get_initial_magnetic_moments()), 3)
        )  # magnetic moments for each atom

        # TODO: document this FSM
        with open(cif_path, mode="r", encoding="utf-8") as cif_f:
            atom_field_order = []
            moment_field_order = []

            cif_f_lines = list(cif_f)
            line_index = 0
            current_mode = "idle"
            while line_index < len(cif_f_lines):
                line = cif_f_lines[line_index]
                if current_mode == "idle":
                    if "data_" in line:
                        name = line.strip().split("_")[-1]
                    elif "_atom_site" in line and not "_atom_site_moment" in line:
                        current_mode = "atom_field_order"
                        continue
                    elif "_atom_site_moment" in line:
                        current_mode = "moment_field_order"
                        continue
                elif current_mode == "atom_field_order":
                    if not "_atom_site" in line or "_atom_site_moment" in line:
                        current_mode = "atoms"
                        continue
                    atom_field_order.append("_".join(line.strip().split("_")[3:]))
                elif current_mode == "atoms":
                    tokens = list(filter(None, line.strip().split(" ")))
                    if len(tokens) != len(atom_field_order):
                        current_mode = "idle"
                        continue
                    for field_index, field in enumerate(atom_field_order):
                        if field == "label":
                            labels.append(tokens[field_index])
                elif current_mode == "moment_field_order":
                    if not "_atom_site_moment" in line:
                        current_mode = "moments"
                        continue
                    moment_field_order.append(line.strip().split(".")[-1])
                elif current_mode == "moments":
                    tokens = list(filter(None, line.strip().split(" ")))
                    if len(tokens) != len(moment_field_order):
                        current_mode = "idle"
                        continue
                    atom_moment = dict()
                    label = None
                    for field_index, field in enumerate(moment_field_order):
                        if field == "label":
                            label = tokens[field_index]
                        elif "crystalaxis_" in field:
                            axis = field.split("_")[-1]
                            atom_moment[axis] = float(tokens[field_index].split("(")[0])

                    moments[labels.index(label)] = [
                        atom_moment["x"],
                        atom_moment["y"],
                        atom_moment["z"],
                    ]

                line_index += 1

        # convert moments to a Cartesian basis
        moments = list(
            map(
                lambda moment: np.sum(
                    np.array([atoms.cell[i] * moment[i] for i in range(3)]), axis=0
                ),
                moments,
            )
        )

        atoms.set_initial_magnetic_moments(moments)
        atoms.set_cell(atoms.cell, scale_atoms=False)
        return cls(name, atoms)

    def get_sf_norm(self, scattering_vector) -> float:
        """Calculates norm of structure factor given a scattering vector

        .. note::
            Implementations of :meth:`~StructureFactor.get_structure_factor()` that return vectors
            must use a Cartesian basis for valid :meth:`~StructureFactor.get_sf_norm()`.

        :param scattering_vector: reciprocal space scattering vector, in reciprocal lattice basis
        :type scattering_vector: tuple[float, float, float]
        :return: norm of the structure factor
        :rtype: float
        """

        return np.linalg.norm(self.get_structure_factor(scattering_vector))

    @abstractmethod
    def get_structure_factor(self, scattering_vector):
        """Abstract method for calculating the structure factor

        :param scattering_vector: reciprocal space scattering vector, in reciprocal lattice basis
        :type scattering_vector: tuple[float, float, float]
        """

    @abstractmethod
    def get_form_factor(self, scattering_vector) -> float:
        """Abstract method dfor calculating the scattering form factor

        :param scattering_vector: reciprocal space scattering vector, in reciprocal lattice basis
        :type scattering_vector: tuple[float, float, float]
        :return: scattering form factor at the given scattering vector
        :rtype: float
        """


class ElasticNeutronFactor(StructureFactor):
    """Implements :class:`StructureFactor` for elastic neutron scattering"""

    def get_structure_factor(self, scattering_vector) -> tuple[float, float, float]:
        """Calculates the elastic neutron scattering structure factor for a given scattering vector

        :param scattering_vector: reciprocal space scattering vector, in reciprocal lattice basis
        :type scattering_vector: tuple[float, float, float]
        :return: complex three-valued elastic neutron scattering structure factor
        :rtype: tuple[float, float, float]
        """
        reciprocal_basis = 2 * np.pi * self.atoms.cell.reciprocal()
        structure_factors = []
        for moment_index, moment in enumerate(
            self.atoms.get_initial_magnetic_moments()
        ):
            pos = self.atoms.get_positions()[moment_index]
            exponent = 1j * sum(
                [
                    pos[i] * scattering_vector[i] * reciprocal_basis[i, i]
                    for i in range(3)
                ]
            )
            structure_factors.append(np.exp(exponent) * moment)

        return tuple(
            self.get_form_factor(scattering_vector)
            * np.sum(np.array(structure_factors), axis=0)
        )

    def get_form_factor(self, scattering_vector):
        """Calculates the neutron magnetic form factor for a given scattering vector

        :param scattering_vector: reciprocal space scattering vector, in reciprocal lattice basis
        :type scattering_vector: tuple[float, float, float]
        :return: neutron magnetic form factor
        :rtype: float
        """
        # TODO: implement more complex form factor varieties
        return 1
