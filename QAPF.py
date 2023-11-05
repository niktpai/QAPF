# %%
"""
Research Project
CS 7375
Nikhil Pai

Implementing an QAPF algorithim to create a path

"""
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy


class MobileRobot():
    """This class represents the mobile robot
    """

    def __init__(self, goal_point=None, start_point=None, map_obstacles=None, map_num=None):
        self.goal_point = goal_point
        self.start_point = start_point
        self.map_obstacles = map_obstacles
        self.map_num = map_num

        self.setEps()
        self.current_state = None
        self.q_table = np.zeros((10, 10, 8))
        self.prob_table = np.zeros(8)
        self.safe = True
        self.action = None

        self.learning_rate = 0.3
        self.discount_factor = 0.8
        self.decision_rate = 0.2
        self.attractive_gain = 0.25
        self.repulsive_gain = 0.60
        self.limit_distance = 4
        self.probabilty = 0.0

        self.path_x = None
        self.path_y = None

        self.reward_good = 100
        self.reward_bad = -1
        self.reward_neutral = 0

        self.save = False
   
    def setEps(self, eps_start_val = 1, to_the_power = 3):
        """Sets the number of episodes ran to generate q-table

        Args:
            eps_start_val (int, optional): (1)e^3. Defaults to 1.
            to_the_power (int, optional): 1e^(3). Defaults to 3.
        """
        self.num_eps = int(eps_start_val * 10 ** to_the_power)
        self.eps_start_val = eps_start_val
        self.to_the_power = to_the_power

    def findNearestNeighboors(self):
        """Finds the locations nearest to the robot in all 8 directions (2d)
        """
        x = 1
        y = 0
        #! could make this for loops
        # self.current_state = np.array([0,5])
        # if current_state is [1,5] then the NW is [0,4], and SE = [2,6]
        top = self.current_state[y] - 1
        bottom = self.current_state[y] + 1
        left = self.current_state[x] - 1
        right = self.current_state[x] + 1

        if top < 0:
            top = 0

        if bottom > 9:
            bottom = 9

        if left < 0:
            left = 0

        if right > 9:
            right = 9

        NW = np.array([top, left])
        N = np.array([top, self.current_state[x]])
        NE = np.array([top, right])

        W = np.array([self.current_state[y], left])
        E = np.array([self.current_state[y], right])

        SW = np.array([bottom, left])
        S = np.array([bottom, self.current_state[x]])
        SE = np.array([bottom, right])

        self.nearest_neighboors = [NW, N, NE, W, E, SW, S, SE]

        # print(self.nearest_neighboors)

    def randomState(self) -> np.array:
        """Returns a random location on the map

        Returns:
            np.array: _description_
        """
        x = np.random.randint(10)
        y = np.random.randint(10)
        return np.array([x, y])

    def computeNearestObs(self, neighboor):
        """Finds the nearest obsticle on the map and returns the distance

        Args:
            neighboor (list): all the nearest locations around the robot

        Returns:
            int: distance to obsticle
        """
        #! could be better optimized
        smallest_dist = 10
        obstacles = np.where(self.map_obstacles == 1)[0]
        for obstacle in obstacles:
            dist = np.linalg.norm(obstacle- neighboor)  
            if dist < smallest_dist:
                smallest_dist = dist
        return smallest_dist

    def computeProbability(self):
        """This will compute all the probabilties on the probability table
        """
        for i in range(len(self.nearest_neighboors)):
            attractive_force = 0.5 * self.attractive_gain * \
                np.linalg.norm(
                    self.nearest_neighboors[i] - self.goal_point) ** 2
            dist_nearest_obs = self.computeNearestObs(
                self.nearest_neighboors[i])

            if (dist_nearest_obs <= self.limit_distance):
                repulsive_force = 0.5 * self.repulsive_gain * (1/dist_nearest_obs -
                                                               1/self.limit_distance) ** 2
            else:
                repulsive_force = 0

            probability = 1 / (attractive_force + repulsive_force)
            self.prob_table[i] = probability

    def computeArtificialPotentialField(self):
        """This will update the self.sigmas will the size of the probabilities relative to the total probability
        """
        self.sigmas = []
        for i in range(len(self.nearest_neighboors)):
            numer = self.prob_table[i]
            denom = sum(self.prob_table)
            self.sigmas.append(numer/denom)

    def chooseAction(self):
        """This function will choose an action
        """
        sorted_sigmas = np.sort(self.sigmas)
        rand_num = np.random.rand()
        for sigma in sorted_sigmas:
            if sigma > rand_num:
                action = np.where(self.sigmas == sigma)[0][0]
                self.action = action
            else:
                #! what if none of sigmas are > rand_num ?
                self.action = np.random.randint(8)

    def bestActionfromQTable(self):
        """Sets action from the best action from the q table
        """
        # self.array = self.q_table[self.current_state[0]][self.current_state[1]]
        best_action = max(self.array)
        self.action = np.where(self.array == best_action)[0][0]

    def performAction(self):
        """Implements the action in the environment
        """
        # self.reward is assigned
        self.future_state = self.nearest_neighboors[self.action]
        if self.map_obstacles[self.future_state[0], self.future_state[1]] == 1:
            self.safe = False

        if (self.future_state[0] == self.goal_point[0] and
                self.future_state[1] == self.goal_point[1]):
            self.reward = self.reward_good
        elif self.safe == False:
            self.reward = self.reward_bad
            # self.safe = True
        else:
            self.reward = self.reward_neutral
        self.newArray = self.q_table[self.future_state[0]
                                     ][self.future_state[1]]

    def findNewState(self):
        """Function will set current_state from the new state
        """

        self.current_state = self.nearest_neighboors[self.action]


    def updateQTable(self):
        """Updates the Q table with estimated cost
        """
        maxQ = max(self.newArray)
        updateValue = (1-self.learning_rate)*self.q_table[self.current_state[0]][self.current_state[1]
                                                                                 ][self.action] + self.learning_rate*(self.reward+self.discount_factor*maxQ)
        self.q_table[self.current_state[0],
                     self.current_state[1], self.action] = updateValue

    def generateQTable(self):
        """Generates Q table by cycling through X episodes
        """
        for ep in range(self.num_eps):
            start = timer()
            self.safe = True
            self.current_state = self.randomState()
            self.array = self.q_table[self.current_state[0]
                                      ][self.current_state[1]]
            while ((self.current_state[0] != self.goal_point[0] or self.current_state[1] != self.goal_point[1]) and self.safe):

                uniform_random_num = np.random.rand()
                self.findNearestNeighboors()
                if (self.decision_rate < uniform_random_num):
                    self.computeProbability()
                    self.computeArtificialPotentialField()
                    self.chooseAction()
                else:
                    uniform_random_num = np.random.rand()
                    if (self.decision_rate < uniform_random_num):
                        self.bestActionfromQTable()
                    else:
                        self.action = np.random.randint(8)
                self.performAction()
                self.updateQTable()
                self.findNewState()
            dt = timer() - start
            # print(f"Time taken {dt}")
            # print(f"Goal or Obstacle reached {self.current_state}")
        if self.save == True:
            file_path = f"qtable_{self.map_num}_{self.eps_start_val}e{self.to_the_power}.mat"
            scipy.io.savemat(file_path, {'q_table' : self.q_table})

    def generatePath(self):
        """Generate path from q table actions
        """
        i = 0
        self.current_state = self.start_point
        self.array = self.q_table[self.current_state[0]][self.current_state[1]]
        self.path_history = []
        self.path_x = []
        self.path_y = []
        self.path_x.append(self.current_state[1])
        self.path_y.append(self.current_state[0])
        self.path_history.append(self.current_state)
        self.safe = True
        while ((self.current_state[0] != self.goal_point[0] or self.current_state[1] != self.goal_point[1]) and self.safe):
            # assuming no environment change for now
            self.findNearestNeighboors()
            self.bestActionfromQTable()
            self.performAction()
            self.findNewState()
            # print(f"At {self.current_state}")
            self.path_x.append(self.current_state[1])
            self.path_y.append(self.current_state[0])
            self.path_history.append(self.current_state)
            self.array = self.q_table[self.current_state[0]
                                      ][self.current_state[1]]
            # print(self.path_history)

    def plotPath(self):
        """" Plots path on 2d plot
        """
        fig, ax = plt.subplots()

        ax.plot(self.path_x, self.path_y)
        ax.plot(self.path_x[0], self.path_y[0], marker='o')
        ax.plot(self.path_x[-1], self.path_y[-1], marker='*')

        for x in range(self.map_obstacles.shape[0]):
            for y in range(self.map_obstacles.shape[1]):
                if self.map_obstacles[x, y] == 1:
                    ax.plot(y, x, marker='X', mec='r', mfc='r')

        ax.set(xlim=(-1, 11), xticks=np.arange(-0.5, 11),
               ylim=(-1, 11), yticks=np.arange(-0.5, 11))

        plt.grid(0.5)
        plt.title(f"Map {self.map_num} || Eps = {self.eps_start_val}e{self.to_the_power}")
        plt.savefig(f"qtable_{self.map_num}_{self.eps_start_val}e{self.to_the_power}")
        print(f"Map {self.map_num} : {self.path_history} ")


    def displayMap(self):
        """
        Just used for testing the display of the map and obsctale
        """
        fig, ax = plt.subplots()

        for x in range(self.map_obstacles.shape[0]):
            for y in range(self.map_obstacles.shape[1]):
                if self.map_obstacles[x, y] == 1:
                    ax.plot(y, x, marker='X', mec='r', mfc='r')

        ax.set(xlim=(-1, 10), xticks=np.arange(-0.5, 10),
        ylim=(-1, 10), yticks=np.arange(-0.5, 10))

        plt.grid(0.5)
        plt.title(f"Map {self.map_num} || Eps = {self.eps_start_val}e{self.to_the_power}")
        plt.savefig(f"qtable_{self.map_num}_{self.eps_start_val}e{self.to_the_power}")
       

# location (height, width)
empty_map = np.zeros((10, 10))

# Map 1
map1 = np.copy(empty_map)
map1[5, 4:7] = 1 # 1 represents an obstacle
goal1 = np.array([1, 5])
start1 = np.array([7, 5]) # (3,5) on the map
myRobot1 = MobileRobot(goal1, start1, map1, 1)
myRobot1.save = True
myRobot1.generateQTable()
myRobot1.generatePath()
myRobot1.plotPath()

# Map 2
map2 = np.copy(empty_map)
map2[6, 0:2] = 1
map2[6, 5:10] = 1
map2[7, 5] = 1
map2[3, 0:2] = 1
map2[3, 3:6] = 1
map2[3, 8:10] = 1
map2[2, 3] = 1
goal2 = np.array([9, 5])
start2 = np.array([2, 8])
myRobot2 = MobileRobot(goal2, start2, map2, 2)
myRobot2.generateQTable()
myRobot2.generatePath()
myRobot2.plotPath()
file_path = f"qtable_{myRobot2.map_num}_{myRobot2.num_eps}.mat"

#Map 3
map3 = np.copy(empty_map)
map3[4:7, 3] = 1
map3[4:7, 6] = 1
map3[4, 3:7] = 1
goal3 = np.array([2, 5])
start3 = np.array([8, 5])
myRobot3 = MobileRobot(goal3, start3, map3, 3)
myRobot3.save = True
myRobot3.generateQTable()
myRobot3.generatePath()
myRobot3.plotPath()


plt.show()

