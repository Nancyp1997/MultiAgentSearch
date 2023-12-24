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
from game import Directions, AgentState

import random, util
import math
from game import Agent
from pacman import GameState
from typing import NamedTuple, Tuple, Union, Any,List, Sequence



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        
        
        # be away from ghost always
        # print(newGhostStates[0].getPosition())
        # print(newGhostStates[0].getDirection())
        foodCnt = successorGameState.getNumFood()
        if foodCnt ==0:
            return math.inf # higher utility val is desirable. Since no food left in grid, we won
        
        ghostDists = [manhattan(ghostLoc.getPosition(), newPos) for ghostLoc in newGhostStates]
        if min(ghostDists) ==0:
            return -math.inf

        food = currentGameState.getFood()
        if food[newPos[0]][newPos[1]]:
            minFoodDist=0
        else:
            foodDists = [
                manhattanDistance(newPos,(x,y))
                for x in range(food.width)
                for y in range(food.height)
                if food[x][y]
            ]
            minFoodDist = min(foodDists)

        # Lesser the food dist, greater the util val. so do 1/minfood,if no food left, 
        # Lesser the ghost dist, greater the penalty
        f_state = (1/(minFoodDist+0.1)) - (1/(min(ghostDists)+0.1))
        # print(f_state)
        # print(successorGameState.getScore())
        # return successorGameState.getScore() + f_state
        return f_state

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def scoreEvaluationFunction(currentGameState: GameState):
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
    pacManIdx = 0

    def getAction(self, gameState: GameState):
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
        
        # Pacman generates its successors    
        pacManActions = gameState.getLegalActions()
        pacManSuccessors = (gameState.generateSuccessor(self.pacManIdx,action) for action in pacManActions)

        # For multi agt envs, we pass them all the pacman's successors. 
        # The adversary would pick the min score of all possible successors it generates from the provisional pacman state 'pacManSucc'
        getAdversariesScores = [self.getMiniMaxScore(pacManSucc,1,self.depth) for pacManSucc in pacManSuccessors]

        # Based on the adversaries scores for each of the pacman's successors, we choose the action that maximises my score
        idx = self.getIndexOfMax(getAdversariesScores) #self.getIndex(max(getAdversariesScores),getAdversariesScores)
        return pacManActions[idx]

    def getIndex(self, score ,scores):
        for i in range(len(scores)):
            if scores[i]==score:
                return i
        return -1



    def getMiniMaxScore(self, gameState: GameState, agentIndex: int, depth: int):
        # If game ended or if we reached end of the state space tree, we return 
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        if agentIndex ==0:
            # Pacman's turn
            nextAgent = 1 # adversary turn after pacman
            nextDepth = depth # for first turn of pacman, depth remains same. for subsequent pacman turns, depth is reduced in logic of adversary.
        else:
            nextAgent = (agentIndex+1)%(gameState.getNumAgents()) # each adversary takes a turn
            # when each of the adversaries take turn, we dont decrement depth as it happens parallely in state space tree
            # If all of the adversaries take a turn based on pacman successor, we decrement depth by 1 once it gets to pacmans turn
            nextDepth = depth - 1 if nextAgent == 0 else depth 
        
        agentActions = gameState.getLegalActions(agentIndex)
        successors = (gameState.generateSuccessor(agentIndex, action) for action in agentActions)
        # Recursively call minimax score function on each of these successors
        miniMaxScores = [ self.getMiniMaxScore(succ, nextAgent, nextDepth) for succ in successors]
        if agentIndex == 0:
            return max(miniMaxScores)
        return min(miniMaxScores)
    
    def getIndexOfMax(self, values: Sequence, default=-1):
        return max(range(len(values)), key=values.__getitem__, default=default)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self._minimax(gameState, 0, self.depth, -math.inf, math.inf)[1]

    def _minimax(self, state, agentIndex: int, depth: int, alpha, beta) -> List:
        if state.isLose() or state.isWin() or depth == 0:
            return [self.evaluationFunction(state),""]        
        # Pacman
        if agentIndex == 0:
            return self.searchMax(state, agentIndex, depth, alpha, beta)
        else:
            return self.searchMin(state, agentIndex, depth, alpha, beta)

    def searchMax(self, state, agentIndex, depth, alpha, beta):
        maxScore = -math.inf
        maxAction = None
        pacManActions = state.getLegalActions(agentIndex)
        for action in pacManActions:
            succ = state.generateSuccessor(agentIndex,action)
            score = self._minimax(succ, 1,depth,alpha,beta)[0]
            if score > beta:
                return [score, action]
            if score > maxScore:
                maxScore = score
                maxAction = action
                if score > alpha:
                    alpha = score
        return [maxScore, maxAction]
    
    def searchMin(self, state, agentIndex, depth, alpha, beta):
        minScore = math.inf
        minAction = None
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if nextAgent>0 else depth-1
        advActions = state.getLegalActions(agentIndex)
        for action in advActions:
            succ = state.generateSuccessor(agentIndex,action)
            
            score = self._minimax(succ, nextAgent, nextDepth, alpha, beta)[0]
            if score < alpha:
                return [score,action]
            if score < minScore:
                minScore = score
                minAction = action
                if score < beta:
                    beta = score
        return [minScore, minAction]




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getIndex(self, score ,scores):
        for i in range(len(scores)):
            if scores[i]==score:
                return i
        return -1
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions()
        succs = [ gameState.generateSuccessor(0,action) for action in actions ]
        scores = [ self._expectimax(succ, 1, self.depth) for succ in succs ]
        i = self.getIndex(max(scores),scores)
        return actions[i]
    
    def _expectimax(self, state, agentIndex, depth):
        if state.isLose() or state.isWin() or depth == 0:
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(agentIndex)
        succs = [ state.generateSuccessor(agentIndex, action) for action in actions ]
        nextAgent = (agentIndex+1)%state.getNumAgents()
        nextDepth = depth if nextAgent > 0 else depth-1
        scores = [ self._expectimax(succ, nextAgent, nextDepth) for succ in succs]
        res = max(scores) if agentIndex == 0 else (sum(scores)/len(actions))
        return res



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentScore = currentGameState.getScore()
    if currentGameState.isWin() or currentGameState.isLose():
        return currentScore
    pacman = currentGameState.getPacmanPosition()
    ghosts: AgentState = currentGameState.getGhostStates()
    ghostDistances = [manhattanDistance(pacman,tuple(map(int, ghost.configuration.pos))) for ghost in ghosts]
    scaredTimers = [ghost.scaredTimer for ghost in ghosts]
    distFromUnscared = [dist for dist, timer in zip(ghostDistances, scaredTimers) if timer == 0]
    distFromScared = [dist for dist, timer in zip(ghostDistances, scaredTimers) if timer > 2]
    ghostPenalty = sum((300 / dist ** 2 for dist in distFromUnscared), 0)
    ghostBonus = sum((190 / dist for dist in distFromScared), 0)
    foods = currentGameState.getFood().asList()
    manhattanDistances = [(manhattanDistance(pacman, food), food) for food in foods]
    manhattanNearestFood = [food for dist, food in sorted(manhattanDistances)[:5]]
    mazeNearestFood = sorted(manhattanDistance(pacman, food) for food in manhattanNearestFood)
    foodBonus = sum(9 / d for d in mazeNearestFood)

    score = currentScore - ghostPenalty + ghostBonus + foodBonus
    return score


    

# Abbreviation
better = betterEvaluationFunction
