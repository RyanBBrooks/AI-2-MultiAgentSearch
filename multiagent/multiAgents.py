# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #minGhostDist = float('inf');
        #for ghostState in newGhostStates:
        #    ghostPos = ghostState.configuration.getPosition()
        #    ghostDist = manhattanDistance(ghostPos,newPos)
        #    if(minGhostDist>ghostDist):
        #        minGhostDist=ghostDist

        minFoodDist = float('inf');
        for food in newFood.asList():
            foodDist = manhattanDistance(food,newPos)
            if(minFoodDist>foodDist):
                minFoodDist=foodDist
        return successorGameState.getScore() + 1.0/(minFoodDist)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def value(gameState,agentIndex,depth):
            #if state is terminal
            if(gameState.isLose() or gameState.isWin() or depth==0):
                #return utility
                return self.evaluationFunction(gameState)
            #if next agent is pacman (MAX) do max-val
            if(agentIndex==0):
                return max_value(gameState,agentIndex,depth)
            #if next agent is ghost (MIN) do min-val
            else:
                return min_value(gameState,agentIndex,depth)

        def max_value(gameState,agentIndex,depth):
            v = float('-inf')

            #determine next agent index
            nextIndex = agentIndex + 1
            #note: agentIntex should always == 0 here, nextIndex should theoretically always be 1 (unless only one agent is pacman)
            if(nextIndex>=gameState.getNumAgents()):
                nextIndex = 0
                # if agent Index = NumAgents-1  which => nextIndex = 0: then depth decrements
                depth-=1

            #return value of the state with max value
            for a in gameState.getLegalActions(agentIndex):
                v = max(v,value(gameState.generateSuccessor(agentIndex,a),nextIndex,depth))
            return v

        def min_value(gameState,agentIndex,depth):
            v = float('inf')

            #determine next agent index
            nextIndex = agentIndex + 1
            #note: agentIntex should always != 0 here
            if(nextIndex>=gameState.getNumAgents()):
                nextIndex = 0
                # if agent Index = NumAgents-1  which => nextIndex = 0: then depth decrements
                depth-=1

            #return value of the state with min value
            for a in gameState.getLegalActions(agentIndex):
                v = min(v,value(gameState.generateSuccessor(agentIndex,a),nextIndex,depth))
            return v

        #decide on which action to take based on minimax values
        maxValue = float('-inf')
        maxAction = None
        for a in gameState.getLegalActions(0):
            #minimax value associated with each action-resulting_state pair for possible initial actions
            v = value(gameState.generateSuccessor(0,a),1,self.depth)
            #update values for new max
            if (v > maxValue):
                maxValue = v
                maxAction = a
        #return action with greatest minimax value
        return maxAction




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    maxAction = ""
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(gameState,agentIndex,depth,a,b):
            #if state is terminal
            if(gameState.isLose() or gameState.isWin() or depth==0):
                #return utility
                return self.evaluationFunction(gameState)
            #if next agent is pacman (MAX) do max-val
            if(agentIndex==0):
                return max_value(gameState,agentIndex,depth,a,b)
            #if next agent is ghost (MIN) do min-val
            else:
                return min_value(gameState,agentIndex,depth,a,b)

        def max_value(gameState,agentIndex,depth,a,b):
            v = float('-inf')

            #determine next agent index
            nextIndex = agentIndex + 1
            #note: agentIntex should always == 0 here, nextIndex should theoretically always be 1 (unless only one agent is pacman)
            if(nextIndex>=gameState.getNumAgents()):
                nextIndex = 0
                # if agent Index = NumAgents-1  which => nextIndex = 0: then depth decrements
                depth-=1
            #return max value of states unless we prune
            for action in gameState.getLegalActions(agentIndex):
                currV=value(gameState.generateSuccessor(agentIndex,action),nextIndex,depth,a,b)
                #if currV is the new max
                if (currV>v):
                    v=currV
                    #if this is the shallowest node save the action
                    if(depth==self.depth):
                        #update our maximizing action
                        self.maxAction = action
                #if greater than or equal to beta, return v
                if (v>b):
                    #if this is the shallowest node save the action
                    if(depth==self.depth):
                        #update our maximizing action
                        self.maxAction = action
                    return v
                #update alpha
                a = max(a,v)
            return v

        def min_value(gameState,agentIndex,depth,a,b):
            v = float('inf')

            #determine next agent index
            nextIndex = agentIndex + 1
            #note: agentIntex should always != 0 here
            if(nextIndex>=gameState.getNumAgents()):
                nextIndex = 0
                # if agent Index = NumAgents-1  which => nextIndex = 0: then depth decrements
                depth-=1

            #return min value of states unless we prune
            for action in gameState.getLegalActions(agentIndex):
                v = min(v,value(gameState.generateSuccessor(agentIndex,action),nextIndex,depth,a,b))
                #if less than or equal to alpha, return v
                if (v<a):
                    return v
                #update beta
                b = min(b,v)
            return v

        value(gameState,0,self.depth,float('-inf'),float('inf'))
        #note because we must start keeping track of a and b from start node:
        #maxAction is now a class var set internally during execution of value.
        return self.maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(gameState,agentIndex,depth):
            #if state is terminal
            if(gameState.isLose() or gameState.isWin() or depth==0):
                #return utility
                return self.evaluationFunction(gameState)
            #if next agent is pacman (MAX) do max-val
            if(agentIndex==0):
                return max_value(gameState,agentIndex,depth)
            #if next agent is ghost (MIN) do min-val
            else:
                return exp_value(gameState,agentIndex,depth)

        def max_value(gameState,agentIndex,depth):
            v = float('-inf')

            #determine next agent index
            nextIndex = agentIndex + 1
            #note: agentIntex should always == 0 here, nextIndex should theoretically always be 1 (unless only one agent is pacman)
            if(nextIndex>=gameState.getNumAgents()):
                nextIndex = 0
                # if agent Index = NumAgents-1  which => nextIndex = 0: then depth decrements
                depth-=1

            #return value of the state with max value
            for a in gameState.getLegalActions(agentIndex):
                v = max(v,value(gameState.generateSuccessor(agentIndex,a),nextIndex,depth))
            return v

        def exp_value(gameState,agentIndex,depth):
            v = 0

            #determine next agent index
            nextIndex = agentIndex + 1
            #note: agentIntex should always != 0 here
            if(nextIndex>=gameState.getNumAgents()):
                nextIndex = 0
                # if agent Index = NumAgents-1  which => nextIndex = 0: then depth decrements
                depth-=1


            actions = gameState.getLegalActions(agentIndex)
            #equal probability for every legal action
            p = 1/len(actions)
            #return state's value times probability
            for a in actions:
                v += p*value(gameState.generateSuccessor(agentIndex,a),nextIndex,depth)
            return v

        #decide on which action to take based on expectimax values
        maxValue = float('-inf')
        maxAction = None
        for a in gameState.getLegalActions(0):
            #expectimax value associated with each action-resulting_state pair for possible initial actions
            v = value(gameState.generateSuccessor(0,a),1,self.depth)
            #update values for new max
            if (v > maxValue):
                maxValue = v
                maxAction = a
        #return action with greatest expectimax value
        return maxAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:

    If the state is a win condition, the value is infinity and thus the best state

    If the state is a lose condition, the value is -infinity and thus the worst state

    Otherwise, the value comprises of the summation of many factors with their
    own weighting. The summed factors and their weights are as follows:

    5(A) + -10(B)*G^(-.5) + 20(C)*G^(-.5) + 7(D) + -35(E) + -500(F)

    SCORE:
    A. (multiple 5) the score of the state is multiplied by 5, the higher the score,
    the better the state is.

    GHOST VARS:
    B. (multiple -10) the inverse of the distance to the closest inedible ghost
    (if there is one). Subtracting this ensures that we maximize the inverse
    distance or in other words minimze the distance between pacman and the ghost.
    This was then multiplied by a factor of the number of ghosts G^-.5 to make it
    less impactful with more ghosts.

    C. (multiple 20) the inverse of the distance to the closest edible ghost (if there is one).
    We want to go towards edible ghosts, so we increase the value of a state more
    if pacman is closer to the edible ghost. Thus he will want to apporach and
    hopefully eat them. This was once again multiplied by a factor of the number
    of ghosts G^-.5 to make it less impactful in the case of more ghosts.

    DOT VARS:
    D. (multiple 7) the inverse of the distance to the closest dot. We
    take the inverse once again to essentially entice Pacman to approach the closest dot.

    E. (multiple -35) number of remaining dots, we want to penalize pacman for
    having more dots on the screen, thus giving him a bonus for moving towards
    the goal of no dots on screen.

    F. (multiple -500) number of remaining capsules (big dots). If he is close to them,
    pacman should really be incentivized to eat these dots as they are worth many points and
    allow him to eat the ghosts thus giving him more points.
    """
    "*** YOUR CODE HERE ***"
    dots = currentGameState.getFood().asList()
    pos = currentGameState.getPacmanPosition()
    value = 0.0

    #CATEGORY: win and loss variables
    if (currentGameState.isWin() or len(dots)==0):
        value = float('inf');
    elif currentGameState.isLose():
        value = float('-inf')
    else:
        value += 5.0 * scoreEvaluationFunction(currentGameState)

        #CATEGORY: ghost related variables
        ghostStates = currentGameState.getGhostStates()
        dstGhostMin = float('inf')
        dstEdibleGhostMin = float('inf')
        isGhost = 0
        isEdibleGhost = 0
        for state in ghostStates:
            ghostPos = state.configuration.getPosition()
            #distance between a ghost and pacman
            dst = manhattanDistance(ghostPos,pos)
            #if the ghost is not edible
            if(state.scaredTimer==0):
                isGhost = 1
                #update the min distance for inedible ghost
                if(dstGhostMin>dst):
                    dstGhostMin=dst
            #if pacman can eat the ghost
            else:
                isEdibleGhost = 1
                #update the min distance for edible ghost
                if(dstEdibleGhostMin>dst):
                    dstEdibleGhostMin=dst
        numGhosts = len(ghostStates)

        #distance to an inedible ghost
        if(isGhost):
            value += (-10.0/dstGhostMin)*pow(numGhosts,(-0.5))
        #distance to an edible ghost
        if(isEdibleGhost):
            value += (20.0/dstEdibleGhostMin)*pow(numGhosts,(-0.5))

        #CATEGORY: dot related variables
        dots = currentGameState.getFood().asList()
        dstFoodMin = float('inf')
        for dotPos in dots:
            #distance between a dot and pacman
            dst = manhattanDistance(dotPos,pos)
            #update the min distance for dot
            if(dstFoodMin>dst):
                dstFoodMin=dst
        #distance to the closest dot
        value += 7.0/dstFoodMin
        #number of dots left
        value += -35.0*len(dots)
        #number of capsules left
        value += -500.0*len(currentGameState.getCapsules())

    return value

# Abbreviation
better = betterEvaluationFunction
