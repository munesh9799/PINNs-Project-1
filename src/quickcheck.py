import numpy as np
from analytic import sdof_response

t = np.linspace(0, 5, 500)
x = sdof_response(t, m=1.0, c=0.2, k=10.0, x0=1.0, v0=0.0)