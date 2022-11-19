from pydnslab.case_setup import case_setup
from pydnslab.model import Model

case = case_setup()
model = Model(case)
model.run()
