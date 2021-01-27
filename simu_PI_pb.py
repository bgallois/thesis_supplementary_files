import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom

'''
Simulation of a Markov chain model with two states 0 and 1.
p the probability 0->1.
b the probability 1->0.
The simulation return the event-based preference index (pi), the ratio exploration-exploitation (r) and the event-based probabilities.
'''


'''
p the probability 0->1.
b the probability 1->0.
N the chain length
'''
def simu(p, b, N):

   # Events as BP is 0->1, PB is 1->2, BB is 0->0 and PP is 1->1
   BP = np.float64(0)
   PB = np.float64(0)
   BB = np.float64(0)
   PP = np.float64(0)

   pos = np.random.randint(0, 2)
   for i in range(int(N)):
     if pos == 0:
      pos = np.random.choice([0,1], p=[1-p, p])
      if pos == 0:
        BB += 1
      else:
        BP += 1
     elif pos == 1:
      pos = np.random.choice([1,0], p=[1-b, b])
      if pos == 1:
        PP += 1
      else:
        PB += 1
    
   P = BP / (BP + BB)
   B = PB / (PB + PP)
   PI = P - B
   R = 2*min(P, B) - 1

   if np.isnan(PI) and np.isnan(P):
     PI = 1-2*B
   elif np.isnan(PI) and np.isnan(B):
     PI = 2*P-1
   return (PI, P, B, R)


'''# Random sampling
N = 200 # Number of points
dist = nbinom(2.526, 0.084).rvs(N) # Based on the experimental distribution
p = np.random.uniform(size=N)
b = np.random.uniform(size=N)

prefIndex = np.zeros(N)
pProba = np.zeros(N)
bProba = np.zeros(N)
ratioE = np.zeros(N)

for i, j in enumerate(dist):
  pi_, p_, b_, r_ = simu(p[i], b[i], j)
  prefIndex[i] = pi_
  pProba[i] = p_
  bProba[i] = b_
  ratioE[i] = r_

# Remove infinite and nan value for impossible Markov chain
delIndex = np.isfinite(prefIndex) & np.isfinite(pProba) & np.isfinite(bProba) & np.isfinite(ratioE)
prefIndex = np.delete(prefIndex, ~delIndex)
pProba = np.delete(pProba, ~delIndex)
bProba = np.delete(bProba, ~delIndex)
ratioE = np.delete(ratioE, ~delIndex)

with open("PI_PB.csv", 'w') as f:
  f.write("pi,p,b,r\n")
  for i, j, k, l in zip(prefIndex, pProba, bProba, ratioE):
    f.write(str(i) +"," + str(j) + "," + str(k) + "," + str(l) + '\n')'''


# Sampling
N = 100 # Number of points
dist = nbinom(2.526, 0.084).rvs(N**2) # Based on the experimental distribution
p = np.linspace(0, 1, N)
b = np.linspace(0, 1, N)

prefIndex = np.zeros(N**2)
pProba = np.zeros(N**2)
bProba = np.zeros(N**2)
ratioE = np.zeros(N**2)
count = 0
for i in range(N):
  for j in range(N):
    pi_, p_, b_, r_ = simu(p[i], b[j], dist[count])
    prefIndex[count] = pi_
    pProba[count] = p_
    bProba[count] = b_
    ratioE[count] = r_
    count += 1

# Remove infinite and nan value for impossible Markov chain
delIndex = np.isfinite(prefIndex) & np.isfinite(pProba) & np.isfinite(bProba) & np.isfinite(ratioE)
prefIndex = np.delete(prefIndex, ~delIndex)
pProba = np.delete(pProba, ~delIndex)
bProba = np.delete(bProba, ~delIndex)
ratioE = np.delete(ratioE, ~delIndex)

with open("PI_PB.csv", 'w') as f:
  f.write("pi,p,b,r\n")
  for i, j, k, l in zip(prefIndex, pProba, bProba, ratioE):
    f.write(str(i) +"," + str(j) + "," + str(k) + "," + str(l) + '\n')

# Visualization
plt.xscale("log")
plt.scatter(ratioE, prefIndex, alpha=0.01)
plt.xlim(0.01,100)
plt.ylim([-1, 1])
plt.xlabel("p/b")
plt.ylabel("Preference Index")

plt.figure()
plt.scatter(pProba, bProba, c=prefIndex)
plt.xlabel("p")
plt.ylabel("b")
clb = plt.colorbar()
clb.ax.set_xlabel("Preference Index")
plt.xlabel("p")
plt.ylabel("b")
plt.show()
