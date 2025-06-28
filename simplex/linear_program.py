import numpy as np

class LinearProgram:
    """
    Representa um problema de programação linear no formato:
        maximize c^T x
        subject to Ax (<=,=,>=) b, x >= 0
    O formato de entrada considera:
    - c: vetor de coeficientes da função objetivo
    - A: matriz de coeficientes das restrições
    - b: vetor de termos independentes
    - signs: lista de sinais das restrições: '<=', '>=', '='
    """

    def __init__(self, c, A, b, signs):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.signs = signs
        self._validate()

    def _validate(self):
        assert self.A.shape[0] == len(self.b) == len(self.signs), \
            "Dimensões de A, b e sinais incompatíveis"
        assert self.A.shape[1] == len(self.c), \
            "Dimensões de A e c incompatíveis"

    @classmethod
    def from_file(cls, path):
        """
        Lê um arquivo texto com o formato:
            m n
            c1 c2 ... cn
            a11 a12 ... a1n sign1 b1
            ...
            am1 am2 ... amn signm bm
        Onde sign é '<=' ou '>=' ou '='.
        """
        with open(path) as f:
            parts = f.read().split()
        it = iter(parts)
        m, n = int(next(it)), int(next(it))
        c = [float(next(it)) for _ in range(n)]
        A, signs, b = [], [], []
        for _ in range(m):
            row = [float(next(it)) for _ in range(n)]
            sign = next(it)
            bi = float(next(it))
            A.append(row)
            signs.append(sign)
            b.append(bi)
        return cls(c, A, b, signs)
