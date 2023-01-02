"""
Upon construction, the Model object will contruct a Fields object,
generate differential operators, 
"""
import numpy as np
import matplotlib.pyplot as plt

import pydnslab.config as config

from pydnslab.grid import Grid
from pydnslab.fields import Fields
from pydnslab.solver import Solver
from pydnslab.operators import Operators
from pydnslab.statistics import Statistics


class Model:
    def __init__(self):

        self.grid: Grid = Grid()
        self.solver = Solver()
        self.operators: Operators = Operators(self.grid)
        self.fields: Fields = Fields(self.grid)
        self.statistics: Statistics = Statistics(self.grid)

    def run(self):

        # Initial projection
        pnew, du, dv, dw = self.solver.projection(self.fields, self.operators)
        self.fields.update(du, dv, dw, pnew)

        # Main time loop
        for i in range(config.nsteps):
            if config.fixed_dt or i == 0:
                dt = config.dt
            else:
                dt = self.solver.adjust_timestep(self.fields, self.grid, dt)

            du, dv, dw = self.solver.timestep(
                self.fields,
                self.operators,
                self.grid,
                dt,
            )

            self.fields.update(du, dv, dw)

            pnew, du, dv, dw = self.solver.projection(self.fields, self.operators)

            self.fields.update(du, dv, dw, pnew)

            self.statistics.update(self.grid, self.fields, i)

        self.statistics.plot()


if __name__ == "__main__":
    model = Model()
    model.run()
