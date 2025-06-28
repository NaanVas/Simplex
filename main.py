import sys
import pandas as pd
from simplex.solver import SimplexSolver

def main():
    if len(sys.argv) != 2:
        print("Uso: python main.py <arquivo_entrada>")
        sys.exit(1)
    filename = sys.argv[1]
    solver = SimplexSolver(filename)
    sol, val, alternatives = solver.solve()

    # DataFrame com a solução ótima
    df = pd.DataFrame({
        "Variável": [f"x{i}" for i in range(1, len(sol)+1)],
        "Valor": sol
    })
    print(f"\nValor ótimo: {val:.4g}")
    print("\nSolução ótima:")
    print(df.to_string(index=False))


    # Exibir outras soluções ótimas, se existirem
    if alternatives:
        print("\nOutras soluções ótimas adjacentes:")
        for idx, alt in enumerate(alternatives, start=1):
            df_alt = pd.DataFrame({
                "Variável": [f"x{i}" for i in range(1, len(alt)+1)],
                "Valor": alt
            })
            print(f"Alternativa #{idx}:")
            print(df_alt.to_string(index=False))
            print("\n")
if __name__ == "__main__":
    main()
