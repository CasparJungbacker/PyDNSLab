"""
Upon construction, the Model object will contruct a Fields object,
generate differential operators, 
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from pydnslab.createfields import Fields
from pydnslab.differentialoperators import Operators


class Model:
    def __init__(self, case: dict):
        self.case: dict = case

        # To be initialized
        self.fields: Fields = None
        self.operators: Operators = None
        self.precon: sps.coo_matrix = None

        # Coefficients for Runge-Kutta time integration
        if case["tim"] == 1:
            self.s = 1
            self.a = np.array([0])
            self.b = 1
            self.c = 0

        elif case["tim"] == 2:
            self.s = 2
            self.a = np.array([[0, 0], [1, 0]])
            self.b = np.array([0.5, 0.5])
            self.c = np.array([0, 1])

        elif case["tim"] == 3:
            self.s = 3
            self.a = np.array([[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]])
            self.b = np.array([1 / 6, 2 / 3, 1 / 6])
            self.c = np.array([0, 0.5, 1])

        elif case["tim"] == 4:
            s = 4
            self.a = np.array(
                [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]
            )
            self.b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
            self.c = np.array([0, 0.5, 0.5, 1])

        self.initialize()

    def initialize(self) -> None:
        self.fields = Fields(self.case)
        self.operators = Operators(self.fields)

        # TODO: petsc
        self.precon = spsl.spilu(self.operators.M)

    def projection(self) -> None:
        div = (
            self.operators.Dx.dot(self.fields.u)
            + self.operators.Dy.dot(self.fields.v)
            + self.operators.Dz.dot(self.fields.w)
        )

        p, exit_code = spsl.bicgstab(
            -self.operators.M, div, x0=self.fields.pold, tol=1e-3, M=self.precon
        )

        self.fields.pold = p

        px = self.operators.Dxp.dot(p)
        py = self.operators.Dyp.dot(p)
        pz = self.operators.Dzp.dot(p)

        self.fields.u -= px
        self.fields.v -= py
        self.fields.w -= pz

    def run(self):
        pass
    