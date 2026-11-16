"""Run attribution-stability (E3) experiments and the Kaikoura transfer map without re-running E1/E2."""
import importlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))
X = importlib.import_module("04_experiments")
if __name__ == "__main__":
    df = X.load()
    imp, stab = X.e3_shap(df); print(stab, flush=True)
    mv = X.e3b_monotonicity_violations(df); print(mv.groupby("model").frac_violation.describe(), flush=True)
    X.predict_kaikoura_map(df); print("kaikoura map done", flush=True)
