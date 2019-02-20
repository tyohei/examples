from generating_data import *
from model_specification import *

# ==== MAP estimate ====
map_estimate = pm.find_MAP(model=basic_model)
print(map_estimate)

map_estimate = pm.find_MAP(model=basic_model, method='powell')
print(map_estimate)


# ==== Sampling methods ====
# - MCMC
#   algorithms
#       - Metropolis
#       - Slice sampling
#       - No-U-Turn Sampler (NUTS)
#       - Hamiltonian MC (HMC)

# -- Gradient-based sampling methods --
