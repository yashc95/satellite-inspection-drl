import util
import numpy as np
import math

def genOnboardPose(ROEpred, depthEst, targetA, targetU):
    def genInertialPositionfromGPS(inertialPosGuess):
        """Generate true inertial position by sampling distribution from guess of inertial position. Guess obtained by
        using predicted ROEs of next state and depth estimate at next state."""
        posGuessMean = inertialPosGuess[0]
        posGuessCov = inertialPosGuess[1]
        truePos = np.random.multivariate_normal(posGuessMean,posGuessCov)
        return truePos
    RTNpred = util.ROE2HILL(ROEpred, targetA, targetU)


#def genObservation():
 #   """Output an estimate of distance b/w camera and target given """

class InspectionMDP(util.MDP):
    def __init__(self, init_state, inspOE_0, targOE_0, RTN_0, dt):
        self.start = list(init_state)
        self.dt = dt + 0
        self.inspOE = np.copy(inspOE_0)
        self.targOE = np.copy(targOE_0)
        theta = math.atan2(RTN_0[1],RTN_0[0])
        phi = math.atan2(RTN_0[2], math.sqrt(math.pow(RTN_0[0],2)+math.pow(RTN_0[1],2)))
        self.Rot= np.matmul(util.rotationMatrix(2,phi),util.rotationMatrix(3,theta))
    # Return the start state.
    def startState(self):
        """In the form [ROE,Geometric position, angular velocity, map, fuel]"""
        return self.start
    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    def succAndProbReward(self, state, action):
        ROE = state[0]
        dROE = util.mapActiontoROE(action, self.inspOE[1], self.targOE[0])
        ROE = ROE + dROE
        newROE = util.propagateROE(ROE, self.dt, self.inspOE[0], self.targOE[0])
        newRTN = util.ROE2HILL(newROE,self.targOE[0],self.targOE[1])
        self.inspOE += util.GVEs(self.inspOE,action,self.dt)
        self.targOE += util.GVEs(self.targOE,np.array([0,0,0]),self.dt)
        fuel_consumed = np.linalg.norm(action)
        newFuel = state[4] - fuel_consumed
        w = state[2]
        R = util.axisAngleRates(w,self.dt)
        self.Rot = np.dot(R,self.Rot)
        newPg = np.dot(self.Rot, np.transpose(newRTN))
        newMap = util.populateMapFeatures(state[3],state[1])
        return [newROE,newPg,w,newMap,newFuel]
    def discount(self):
        return 1

