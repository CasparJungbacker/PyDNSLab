"""
Upon construction, the Model object will contruct a Fields object,
generate differential operators, 
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from pydnslab.createfields import Fields
from pydnslab.scipy_operators import ScipyOperators
from pydnslab.solver import solver
from pydnslab.projection import projection
from pydnslab.adjust_timestep import adjust_timestep


class Model:
    def __init__(self, case: dict):
        self.case: dict = case

        # To be initialized
        self.fields: Fields = None
        self.operators: ScipyOperators = None
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
            self.s = 4
            self.a = np.array(
                [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]
            )
            self.b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
            self.c = np.array([0, 0.5, 0.5, 1])

        self.initialize()

    def initialize(self) -> None:
        self.fields = Fields(self.case)
        self.operators = ScipyOperators(self.fields)

    def run(self):

        # Initial projection
        projection(self.fields, self.operators)

        # Main time loop
        for i in range(self.case["nsteps"]):
            if self.case["fixed_dt"]:
                dt = self.case["dt"]
            else:
                dt = adjust_timestep(
                    self.fields, self.case["dt"], self.case["co_target"]
                )
            solver(
                self.fields,
                self.operators,
                self.s,
                self.a,
                self.b,
                self.c,
                dt,
                self.case["nu"],
                self.case["gx"],
                self.case["gy"],
                self.case["gz"],
            )

            if i % 100 == 0:
                print(f"step: {i}")
