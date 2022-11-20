"""
Upon construction, the Model object will contruct a Fields object,
generate differential operators, 
"""
import numpy as np
import matplotlib.pyplot as plt

from pydnslab import butcher_tableau
from pydnslab.grid import Grid
from pydnslab.fields.get_fields import get_fields
from pydnslab.solver.get_solver import get_solver
from pydnslab.operators.get_operators import get_operators
from pydnslab.fields.basefields import Fields
from pydnslab.operators.base_operators import Operators
from pydnslab.solver.basesolver import Solver
from pydnslab.statistics import Statistics


class Model:
    def __init__(self, case: dict):
        self.settings: dict = case

        self.grid: Grid = Grid(
            self.settings["res"], self.settings["l_scale"], self.settings["w_scale"]
        )

        self.operators: Operators = get_operators(self.grid, self.settings["engine"])

        self.fields: Fields = get_fields(
            self.grid.griddim,
            self.settings["runmode"],
            self.settings["u_nom"],
            self.settings["u_f"],
            self.settings["engine"],
        )

        self.solver: Solver = get_solver(self.settings["engine"])

        self.statistics: Statistics = Statistics(self.grid, self.settings)

        s, a, b, c = butcher_tableau(self.settings["tim"])
        self.s: int = s
        self.a: np.ndarray = a
        self.b: np.ndarray = b
        self.c: np.ndarray = c

    def run(self):

        # Initial projection
        pnew, du, dv, dw = self.solver.projection(self.fields, self.operators)
        self.fields.update(du, dv, dw, pnew)

        # Main time loop
        for i in range(self.settings["nsteps"]):
            if self.settings["fixed_dt"]:
                dt = self.settings["dt"]
            else:
                dt = self.solver.adjust_timestep(
                    self.fields,
                    self.grid,
                    self.settings["dt"],
                    self.settings["co_target"],
                )

            du, dv, dw = self.solver.timestep(
                self.fields,
                self.operators,
                self.grid,
                self.s,
                self.a,
                self.b,
                self.c,
                dt,
                self.settings["nu"],
                self.settings["gx"],
                self.settings["gy"],
                self.settings["gz"],
            )

            self.fields.update(du, dv, dw)

            pnew, du, dv, dw = self.solver.projection(self.fields, self.operators)

            self.fields.update(du, dv, dw, pnew)

            self.statistics.update(self.grid, self.fields)

            if i % 100 == 0:
                print(i)

        fig, ax = plt.subplots()
        ax.plot(self.statistics.yplumean, self.statistics.uplumean)
        plt.show()
