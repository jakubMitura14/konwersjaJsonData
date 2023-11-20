import numpy as np


def test_get_max_displacements():
    gold = np.zeros((5, 5, 5), dtype=bool)
    gold[1, 1, 1] = True
    algo = np.zeros((5, 5, 5), dtype=bool)
    algo[0, 0, 0] = True
    algo[2, 3,4] = True
    x_diff, y_diff, z_diff = get_max_displacements(gold, algo)
    print(f"x_diff {x_diff} y_diff {y_diff} z_diff {z_diff}")
    assert x_diff == 1
    assert y_diff == 2
    assert z_diff == 3

