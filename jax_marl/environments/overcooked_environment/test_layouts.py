from jax_marl.environments.overcooked_environment.layouts import evaluate_grid
from jax_marl.environments.overcooked_environment import overcooked_layouts

def test_grid_not_enough_elements():
    grid = """
    WWWWWWW
    W  P  W
    W A A W
    WO   BW
    W     W
    WWWWWWW
    """
    assert evaluate_grid(grid) == False

def test_grid_unreachable_elements():
    grid = """
    WWWWWWW
    W  P  W
    WWWWWWW
    W A A W
    WO   BW
    W  X  W
    WWWWWWW
    """
    assert evaluate_grid(grid) == False

def test_grid_not_enclosed():
    grid = """
    WWWWWWW
    W  P  W
    W A A W
    WO   BW
    W  X  W
    """
    assert evaluate_grid(grid) == False

def test_grid_enclosed_elements():
    grid = """
    WWWWWWW
    W  P  W
    W A A W
    WO W BW
    W WXW W
    WWWWWWW
    """
    assert evaluate_grid(grid) == False

def test_grid_no_shared_wall():
    grid = """
    WWWWWWWWWWWWWWW
    W  P  W W  P  W
    W A   W W A   W
    WO   BW WO   BW
    W  X  W W     W
    WWWWWWWWWWWWWWW
    """
    assert evaluate_grid(grid) == False

def test_valid_grid():
    grid = """
    WWWWWWW
    W  P  W
    W A A W
    WO   BW
    W  X  W
    WWWWWWW
    """
    assert evaluate_grid(grid) == True
