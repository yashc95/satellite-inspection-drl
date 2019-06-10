import mdp
import util
import math
import numpy as np
import matplotlib.pyplot as plt

#OE's in form [a, u, ex, ey, i, RAAN]. Angles in radians
oe_insp_0 = np.array([6771.0, 0.0, 0.03, 0.0, math.radians(5.0), 0.0])
oe_targ_0 = np.array([6771.0, 0.0, 0.0, 0.0, math.radians(5.0), 0.0])
roe_0 = util.ROEfromOEs(oe_insp_0,oe_targ_0)
RTN_0 = util.ROE2HILL(roe_0,oe_targ_0[0],oe_targ_0[1])
w = 0.0
w_0 = np.array([0, 0, w])
Pg_0 = np.array([np.linalg.norm(RTN_0), 0, 0])
# Array to store "feature points" by theta, phi pair in Geometric frame.
map_0 = np.zeros((360,360))
fuel_0 = 50
state_0 = [roe_0,Pg_0,w_0,map_0,fuel_0]
dt = 100
iterations = 100000


"""Baseline"""
Insp_wcalc = mdp.InspectionMDP(state_0, oe_insp_0, oe_targ_0, RTN_0, dt)
state = Insp_wcalc.startState()
action = np.array([0,0,0])
w_track = []
Pg_baseline_hist = []
Pg_baseline_hist.append(tuple(state[1]))
for i in range(iterations):
    state = Insp_wcalc.succAndProbReward(state,action)
    Pg_baseline_hist.append(tuple(state[1]))
    vec2 = state[1]
    vec1 = Pg_baseline_hist[i]
    angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    w_i = [0,0, -1*float(angle)/dt]
    w_track.append(tuple(w_i))
map_0 = np.zeros((360,360))
state_0 = [roe_0,Pg_0,w_0,map_0,fuel_0]
Insp = mdp.InspectionMDP(state_0, oe_insp_0, oe_targ_0, RTN_0, dt)
state = Insp.startState()
state[2] = w_track[0]
action = np.array([0,0,0])
Pg_baseline_hist = []
Pg_baseline_hist.append(tuple(state[1]))
for i in range(iterations):
    state = Insp.succAndProbReward(state,action)
    Pg_baseline_hist.append(tuple(state[1]))
    if i != iterations-1:
        state[2] = w_track[i+1]
map_baseline = state[3]
fuel = state[4]
percent_coverage = float(np.count_nonzero(map_baseline))/map_baseline.size
print("Baseline Fuel Remaining: ", fuel)
print("Baseline Percent Coverage: ", percent_coverage)
"""Oracle"""
map_0 = np.zeros((360,360))
state_0 = [roe_0,Pg_0,w_0,map_0,fuel_0]
Insp2 = mdp.InspectionMDP(state_0, oe_insp_0, oe_targ_0, RTN_0, dt)
action_oracle = np.array([0,0,0])
state_oracle = Insp2.startState()
Pg_oracle_hist = []
Pg_oracle_hist.append(tuple(state_oracle[1]))
for i in range(iterations):
    state_oracle = Insp2.succAndProbReward(state_oracle, action_oracle)
    Pg_oracle_hist.append(tuple(state_oracle[1]))
map_oracle = state_oracle[3]
fuel_oracle = state_oracle[4]
percent_coverage_oracle = float(np.count_nonzero(map_oracle))/map_oracle.size
print("Oracle Fuel Remaining: ", fuel_oracle)
print("Oracle Percent Coverage: ", percent_coverage_oracle)
"""Plot Baseline"""
x_b = [Pg[0] for Pg in Pg_baseline_hist]
y_b = [Pg[1] for Pg in Pg_baseline_hist]
z_b = [Pg[2] for Pg in Pg_baseline_hist]
x_o = [Pg[0] for Pg in Pg_oracle_hist]
y_o = [Pg[1] for Pg in Pg_oracle_hist]
z_o = [Pg[2] for Pg in Pg_oracle_hist]
plt.figure(1)
plt.plot(x_b,y_b)
ax1 = plt.gca()
ax1.set_xlim(-500,500)
ax1.set_ylim(-500,500)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_title('Baseline')
plt.figure(2)
plt.plot(x_o,y_o)
ax2 = plt.gca()
ax2.set_xlim(-500,500)
ax2.set_ylim(-500,500)
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_title('Oracle')
plt.show()
print("Wait")