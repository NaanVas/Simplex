from .linear_program import LinearProgram
from .tableau import SimplexTableau

class SimplexSolver:
    """
    Interface de execução do algoritmo Simplex.
    """
    def __init__(self, filename):
        self.lp = LinearProgram.from_file(filename)
        self.tableau = SimplexTableau(self.lp)

    def solve(self):
        sol, val, alternatives = self.tableau.optimize()
        return sol, val, alternatives
