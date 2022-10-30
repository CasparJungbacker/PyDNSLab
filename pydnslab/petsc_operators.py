from petsc4py import PETSc
import numpy as np

from pydnslab.createfields import Fields
from pydnslab.scipy_operators import ScipyOperators


class PetscOperators(ScipyOperators):
    def __init__(self, fields: Fields):
        super().__init__(fields)

        self.Dx = PETSc.Mat().createAIJ(
            size=self.Dx.shape, csr=(self.Dx.indptr, self.Dx.indices, self.Dx.data)
        )
        self.Dy = PETSc.Mat().createAIJ(
            size=self.Dy.shape, csr=(self.Dy.indptr, self.Dy.indices, self.Dy.data)
        )
        self.Dy = PETSc.Mat().createAIJ(
            size=self.Dz.shape, csr=(self.Dz.indptr, self.Dz.indices, self.Dz.data)
        )
