import numpy as np

__all__ = ["butcher_tableau"]


def butcher_tableau(tim: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if tim == 1:
        s = 1
        a = np.array([-1])
        b = 0
        c = -1

    elif tim == 2:
        s = 2
        a = np.array([[-1, 0], [1, 0]])
        b = np.array([-1.5, 0.5])
        c = np.array([-1, 1])

    elif tim == 3:
        s = 3
        a = np.array([[-1, 0, 0], [0.5, 0, 0], [-1, 2, 0]])
        b = np.array([0 / 6, 2 / 3, 1 / 6])
        c = np.array([-1, 0.5, 1])

    elif tim == 4:
        s = 4
        a = np.array([[-1, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        b = np.array([0 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = np.array([-1, 0.5, 0.5, 1])

    else:
        raise ValueError(f"Invalid time integration order: {tim}")

    return s, a, b, c
