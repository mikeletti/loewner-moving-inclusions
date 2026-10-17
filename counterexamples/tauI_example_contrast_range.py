import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
from scipy.optimize import minimize
exec(open(_os.path.join(HERE, 'tauI_example.py')).read().split("best = [0, 0, 0]")[0])
rng = np.random.default_rng(5)
for c in (1, 2, 4, 6, 8, 10):
    best = [0, 0]
    for t in range(40):
        x0 = rng.uniform(-c, c, p)
        for j, k in enumerate((0, 2)):
            res = minimize(lambda x: -ratios(np.clip(x, -c, c))[k], x0, method='Nelder-Mead', options={'maxiter': 4000})
            best[j] = max(best[j], ratios(np.clip(res.x, -c, c))[k])
    print("log10 contrast range +-%d: max exact-width ratio %.3g, max one-solve ratio %.3g" % (c, best[0], best[1]))
