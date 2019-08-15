import gym
from gym import error, spaces, utils
from gym.utils import seeding
import util
import numpy as np
import math


class SatInspectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_state, map, inspOE_0, targOE_0, RTN_0, dt, num_photos, tau):
        self.init_state = list(init_state)
        self.state = list(init_state)
        self.RTNlist = [RTN_0]
        self.RTNlistbest = []
        self.map_0 = np.copy(map)
        self.map = np.copy(map)
        self.dt = dt + 0
        self.inspOE = np.copy(inspOE_0)
        self.targOE = np.copy(targOE_0)
        self.inspOE_0 = np.copy(inspOE_0)
        self.targOE_0 = np.copy(targOE_0)
        self.RTN_0 = np.copy(RTN_0)
        self.num_photos = num_photos
        self.photos_left = num_photos
        theta = math.atan2(RTN_0[1], RTN_0[0])
        phi = math.atan2(RTN_0[2], math.sqrt(math.pow(RTN_0[0], 2) + math.pow(RTN_0[1], 2)))
        self.Rot = np.matmul(util.rotationMatrix(2, phi), util.rotationMatrix(3, theta))
        self.tau = tau
        self.done = 0
        self.reward = 0
        self.best_reward = 0
        self.min_action = np.array([-1.0, -1.0, -1.0, -1.0])
        self.max_action = np.array([1.0, 1.0, 1.0, 1.0])
        self.no_image_count = 0
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)
        """TO TRY:
        ACTIONS = Mag plus directionality
        or ACTIONS = binary (0,1) and thrust vector"""

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state
        ind = action[0]
        """
        if ind<0:
            dROE = np.array([0, 0, 0, 0, 0, 0])
            dROE[1]= math.sqrt(mu)*(math.pow(deputyA,-1.5)-math.pow(chiefA,-1.5))
        else:
            dROE = action[1:5]
        """
        if ind < 0:
            true_action = 0.0 * action[1:4]
        else:
            true_action = ind * action[1:4]
        true_action = [a / float(1000) for a in true_action]
        img_count = 0
        ROE = state[0:6]
        w = state[9:12]
        oldWhereCovered = self.map[np.where(self.map >= 2)]
        oldRTN = util.ROE2HILL(ROE, self.targOE[0], self.targOE[1])
        dROE = util.mapActiontoROE(true_action, self.inspOE[1], self.targOE[0])
        ROE = ROE + dROE
        newMap = np.copy(self.map)
        for i in range(0, self.dt, self.tau):
            if i == 0:
                self.inspOE = util.GVEs(self.inspOE, true_action, 1)
                self.targOE = util.GVEs(self.targOE, np.array([0, 0, 0]), 1)
                newRTN = util.ROE2HILL(ROE, self.targOE[0], self.targOE[1])
                if np.linalg.norm(newRTN) <= 0.1:
                    newMap = util.populateMapFeatures(self.map, state[6:9])
                    img_count += 1
            else:
                self.inspOE = util.GVEs(self.inspOE, np.array([0, 0, 0]), self.tau)
                self.targOE = util.GVEs(self.targOE, np.array([0, 0, 0]), self.tau)
                newROE = util.propagateROE(ROE, self.tau, self.inspOE[0], self.targOE[0])
                newRTN = util.ROE2HILL(newROE, self.targOE[0], self.targOE[1])
                R = util.axisAngleRates(w, self.tau)
                self.Rot = np.dot(R, self.Rot)
                newPg = np.dot(self.Rot, np.transpose(newRTN))
                if np.linalg.norm(newRTN) <= 0.1:
                    newMap = util.populateMapFeatures(newMap, newPg)
                    img_count += 1

        fuel_consumed = np.linalg.norm(true_action)
        newFuel = state[12] - fuel_consumed
        newFuelvec = np.array((newFuel,))
        if img_count == 0:
            self.no_image_count += 1
        else:
            self.no_image_count = 0
        self.state = np.concatenate((newROE, newPg, w, newFuelvec), axis=0)
        self.photos_left += -1
        whereCovered = newMap[np.where(newMap >= 2)]
        self.percent_coverage = float(whereCovered.size) / self.map.size
        newFeatures = whereCovered.size - oldWhereCovered.size
        self.map = np.copy(newMap)
        if np.linalg.norm(newRTN) < np.linalg.norm(oldRTN):
            self.reward += 1
        if self.percent_coverage > 0.70:
            self.done = 1
            self.reward += 10000
            print('Done because finished map')
        else:
            if img_count > 0 and newFeatures <= 200:
                print('Done because not enough new features seen')
                self.reward += -2000
                self.done = 1
        if newFuel <= 0:
            self.done = 1
            print('Done because no more fuel')

        if self.photos_left == 1:
            self.done = 1
            print('Done because no photos left')
        if self.no_image_count >= 10:
            self.done = 1
            self.reward += -500
            print('Done because did not see target for a long time')
        if np.linalg.norm(newRTN) <= 0.001:
            self.done = 1
            self.reward += -10000
            print('Done because hit target')
        if self.done == 1:
            if self.percent_coverage > 0.50:
                self.reward += self.percent_coverage * 100000
            elif self.percent_coverage < 0.1:
                self.reward += -self.percent_coverage * 10000
            else:
                self.reward += self.percent_coverage * 10000
            print('Final Percent Coverage: ', self.percent_coverage)
        print('Actions Left', self.photos_left, 'RTN: ', newRTN, 'Fuel Consumed', fuel_consumed,
              'New Features', newFeatures)
        self.RTNlist.append(newRTN)
        return [self.state, self.reward, self.done, self.map]

    def reset(self):
        self.photos_left = self.num_photos
        self.state = list(self.init_state)
        """
        w0 = self.init_state[9:12]
        cov = (np.linalg.norm(w0))*np.eye(3)
        w = np.random.multivariate_normal(w0, cov, 1)
        print("W = ",w)
        self.state[9:12] = list(w[0])
        """
        self.inspOE = np.copy(self.inspOE_0)
        self.targOE = np.copy(self.targOE_0)
        theta = math.atan2(self.RTN_0[1], self.RTN_0[0])
        phi = math.atan2(self.RTN_0[2], math.sqrt(math.pow(self.RTN_0[0], 2) + math.pow(self.RTN_0[1], 2)))
        self.Rot = np.matmul(util.rotationMatrix(2, phi), util.rotationMatrix(3, theta))
        self.map = np.copy(self.map_0)
        self.done = 0
        if self.reward > self.best_reward:
            self.best_reward = self.reward
            self.RTNlistbest = self.RTNlist
        else:
            self.RTNlist = [self.RTN_0]
        self.reward = 0
        return self.state
    
    def getBestRun(self):
        return self.RTNlistbest, self.best_reward
    def render(self):
        print(self.map)