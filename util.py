import math
import numpy as np


class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)
        # print self.states

def propagateROE(ROE,dt, deputyA, chiefA):
    mu = 398600
    #For now just use central body gravity. Incorporate J2 later.
    dROE = np.array([0, 0, 0, 0, 0, 0])
    dROE[1]= math.sqrt(mu)*(math.pow(deputyA,-1.5)-math.pow(chiefA,-1.5))
    newROE = ROE + dROE*dt
    return newROE


def ROE2HILL(ROE, targetA, targetU):
    """
    :param ROEs: A list of relative orbital elements
    :param targetA: semi-major axis of target
    :param targetU: mean argument of latitude for target
    :return: Hill coordinates of satellite relative to the "chief", in this case the "target"
    """
    da = ROE[0]
    dl = ROE[1]
    dex = ROE[2]
    dey = ROE[3]
    dix = ROE[4]
    diy = ROE[5]
    a = targetA
    u = targetU
    drr = a*(da-dex*math.cos(u)-dey*math.sin(u))
    drt = a*(-1.5*da*u+dl+2*dex*math.sin(u)-2*dey*math.cos(u))
    drn = a*(dix*math.sin(u)-diy*math.cos(u))
    RTN = (drr, drt, drn)
    return np.array(RTN)

def ROEfromOEs(deputy_oe,chief_oe):
    """
    Inputs absolute OEs in form [a, u, ex, ey, i, RAAN] where u is the mean argument of latitude and
    outputs the ROEs
    :param deputy_oe:
    :param chief_oe:
    :return: Quasi-nonsingular ROEs
    """
    da = (deputy_oe[0]-chief_oe[0])/chief_oe[0]
    dl = (deputy_oe[1]-chief_oe[1])+(deputy_oe[5]-chief_oe[5])*math.cos(chief_oe[4])
    dex = deputy_oe[2]-chief_oe[2]
    dey = deputy_oe[3]-chief_oe[3]
    dix = deputy_oe[4]-chief_oe[4]
    diy = (deputy_oe[5]-chief_oe[5])*math.sin(chief_oe[4])
    return np.array([da, dl, dex, dey, dix, diy])

def GVEs(oe, delta_v,dt):
    """
    Computes change in absolute OEs given an RTN delta-v vector
    :param oe: [a, u, ex, ey, i, RAAN]
    :param delta_v: [vr,vt,vn]
    :return:
    """
    mu = 398600
    a = oe[0]
    u = oe[1]
    i = oe[4]
    if a < 0:
        print ('oh no')
    n = math.sqrt(float(mu)/math.pow(a,3))
    da = (2/n)*delta_v[1]
    du = n + (1/(n*a))*(-2*delta_v[0]-math.sin(u)*math.cos(i)/math.sin(i)*delta_v[2])
    dex = (1/(n*a))*(math.sin(u)*delta_v[0]+2*math.cos(u)*delta_v[1])
    dey = (1/(n*a))*(-math.cos(u)*delta_v[0]+2*math.sin(u)*delta_v[1])
    di = (1/(n*a))*math.cos(u)*delta_v[2]
    dRAAN = (1/(n*a))*math.sin(u)/math.sin(i)*delta_v[2]
    oe = oe + [dt*el for el in [da,du,dex,dey,di,dRAAN]]
    return oe


def mapActiontoROE(action,deputyU,deputyA):
    a = deputyA
    mu = 398600
    n = math.sqrt(float(mu)/math.pow(a,3))
    u = deputyU
    gamma = (1/(n*a))*np.array([[0,2,0],[-2,0,0],[math.sin(u),2*math.cos(u),0],[-math.cos(u),2*math.sin(u),0],[0,0,math.cos(u)],[0,0,math.sin(u)]])
    dROE = np.matmul(gamma,action)
    return dROE

def rotationMatrix(axis,theta):
    if axis == 1:
        R = np.array([[1,0,0],[0, math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]])
    elif axis == 2:
        R = np.array([[math.cos(theta),0,-math.sin(theta)],[0, 1,0],[math.sin(theta),0,math.cos(theta)]])
    elif axis == 3:
        R = np.array([[math.cos(theta),math.sin(theta),0],[-math.sin(theta), math.cos(theta),0],[0,0,1]])
    return R


def axisAngleRates(w,dt):
    angular_speed = np.linalg.norm(w)
    theta = angular_speed*dt
    if angular_speed == 0:
        R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    else:
        axis = (1/angular_speed)*np.array(w)
        R1 = math.cos(theta)*np.identity(3)
        R2 = np.array((1-math.cos(theta))*np.matmul(np.transpose(np.asmatrix(axis)),np.asmatrix(axis)))
        ax = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.array(np.linalg.inv(R1 + R2 + math.sin(theta)*ax))
    return R

def populateMapFeatures(map,Pg):
    theta = math.degrees(math.atan2(Pg[1],Pg[0]))
    phi = math.degrees(math.atan2(Pg[2], math.sqrt(math.pow(Pg[0], 2) + math.pow(Pg[1], 2))))
    mean = np.array([theta,phi])
    cov = np.array([[1000, 0],[0,1000]])
    features = np.random.multivariate_normal(mean,cov,2000)
    newmap = np.copy(map)
    for i in range(np.shape(features)[0]):
        features[i] = np.rint(features[i])
        if features[i,0] < 0:
            features[i,0] += 360
        if features[i,1] < 0:
            features[i,1] += 360
        if features[i,0] > 359:
            features[i,0] += -360
        if features[i,1] > 359:
            features[i,1] += -360
        newmap[int(features[i,0]),int(features[i,1])] += 1
    return newmap
