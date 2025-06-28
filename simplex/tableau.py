import numpy as np
import copy
from .linear_program import LinearProgram

class SimplexTableau:
    """
    Monta e manipula o tableau do Simplex, incluindo fase I e II.
    """
    def __init__(self, lp: LinearProgram):
        self.lp = lp
        self._build_initial_tableau()

    def _build_initial_tableau(self):
        # Padronizar restrições e adicionar variáveis de folga/excesso e artificiais
        A, b, signs = self.lp.A.copy(), self.lp.b.copy(), self.lp.signs
        m, n = A.shape
        tableaux = []
        slack = []
        art = []
        col = n
        for i, sign in enumerate(signs):
            if sign == '<=':
                A = np.hstack([A, np.eye(m)[:, i:i+1]])
                slack.append(col)
                col += 1
                art.append(None)
            elif sign == '>=':
                # excesso e artifical
                excess = -np.eye(m)[:, i:i+1]
                A = np.hstack([A, excess])
                slack.append(col)
                col += 1
                # artificial
                A = np.hstack([A, np.eye(m)[:, i:i+1]])
                art.append(col)
                col += 1
            else:  # '='
                A = np.hstack([A, np.zeros((m,1))])
                slack.append(None)
                # artificial
                A = np.hstack([A, np.eye(m)[:, i:i+1]])
                art.append(col)
                col += 1
        # construir tableau
        self.table = np.zeros((m+1, col+1))
        self.table[:m, :col] = A
        self.table[:m, -1] = b
        # função objetivo
        self.table[-1, :n] = -self.lp.c
        self.basic = []
        # definir base inicial
        for i in range(m):
            # se há artificial, usa artificial, senão slack
            if art[i] is not None:
                self.basic.append(art[i])
                # ajustar fase I
            else:
                self.basic.append(n + i)
        self._init_phase1_objective(art)

    def _init_phase1_objective(self, art):
        # construir função objetivo da fase I: minimizar soma de artificiais
        m, col = self.table.shape[0]-1, self.table.shape[1]-1
        for i, ai in enumerate(art):
            if ai is not None:
                # somar linha i à linha objetivo fase I
                self.table[-1, :] += self.table[i, :]
        self.phase = 1

    def _choose_entering(self):
        # variável com custo reduzido negativo
        costs = self.table[-1, :-1]
        idxs = np.where(costs < -1e-8)[0]
        return idxs[0] if len(idxs) else None

    def _choose_leaving(self, entering):
        # razão mínima b_i / a_i_entering, a_i_entering > 0
        ratios = []
        for i in range(len(self.basic)):
            a = self.table[i, entering]
            if a > 1e-8:
                ratios.append((self.table[i, -1] / a, i))
        if not ratios:
            return None
        # critério Bland ou outro
        _, row = min(ratios, key=lambda x: x[0])
        return row

    def _pivot(self, row, col):
        # pivotamento: tornar pivot = 1 e zerar coluna
        self.table[row, :] /= self.table[row, col]
        for i in range(self.table.shape[0]):
            if i != row:
                self.table[i, :] -= self.table[i, col] * self.table[row, :]
        self.basic[row] = col

    def optimize(self):
        # executa fases até solução ótima
        while True:
            entering = self._choose_entering()
            if entering is None:
                break
            leaving = self._choose_leaving(entering)
            if leaving is None:
                raise Exception("Problema ilimitado")
            self._pivot(leaving, entering)
        # trocar para fase II se fase I
        if self.phase == 1:
            self._prepare_phase2()
            return self.optimize()
        # extração da solução principal
        sol, val = self._extract_solution()
        # busca de soluções alternativas
        alternatives = self.find_alternatives()
        return sol, val, alternatives

    def find_alternatives(self):
        """
        Retorna lista de outras soluções BFS ótimas adjacentes
        pivotando em variáveis não-básicas com custo reduzido = 0
        """
        alts = []
        cost_row = self.table[-1, :-1]
        nonbasics = [j for j in range(cost_row.size) if j not in self.basic]
        for j in nonbasics:
            if abs(cost_row[j]) < 1e-8:
                # cópia profunda do tableau
                t = copy.deepcopy(self)
                row = t._choose_leaving(j)
                if row is not None:
                    t._pivot(row, j)
                    x, _ = t._extract_solution()
                    alts.append(x)
        return alts

    def _prepare_phase2(self):
        # elimina função fase I, define objetivo original, remove artificiais
        n = len(self.lp.c)
        self.table[-1, :] = 0
        self.table[-1, :n] = -self.lp.c
        self.phase = 2

    def _extract_solution(self):
        n = len(self.lp.c)
        x = np.zeros(n)
        for i, bi in enumerate(self.basic):
            if bi < n:
                x[bi] = self.table[i, -1]
        value = self.table[-1, -1]
        return x, value
