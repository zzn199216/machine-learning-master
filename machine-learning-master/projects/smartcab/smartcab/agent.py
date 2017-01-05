import random
from numpy import arange
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha=.5, gamma=.2,epsilon=.1):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.state = None
        self.state_for_train = None
        self.count_reward=0
        self.qlearn = Qlearner( alpha, gamma ,epsilon)

        self.turn = 0 #the turn of this Learning now
        self.num_of_learn = 0 #the number of times of learn now
        self.learns = {} #get state of every turns

        self.go_learn = True #set if should use qlearn


    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.turn = 0
        self.count_reward=0
        self.num_of_learn += 1
        # TODO: Prepare for a new trip; reset any variables here, if required

        print '*****',self.learns



    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs, self.next_waypoint)
        should_move = True

        if self.next_waypoint == 'left' and (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
            should_move = False

        if inputs['light'] == 'red':
            if self.next_waypoint == 'straight' or self.next_waypoint == 'left':
                should_move = False
            elif self.next_waypoint == 'right' and inputs['left'] == 'forward':
                should_move = False

        # TODO: Select action according to your policy
        action = None
        # Execute action and get reward
        # reward = self.env.act(self, action)
        # self.count_reward += reward


        # TODO: Learn policy based on state, action, reward
        if self.go_learn:
            #print '**************1',inputs
            self.state_for_train = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

            #learn a action use Qlearn
            action = self.qlearn.select_action(self.state_for_train)

            reward = self.env.act(self, action)

            next_inputs = self.env.sense(self)
            next_state =  (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], self.next_waypoint)
            # print 'state now:',self.state_for_train ,'next state:',next_state
            self.qlearn.learn(self.state_for_train, next_state, action, reward)

        else:
            action = random.choice(Environment.valid_actions)

        reward = self.env.act(self, action)
        self.turn += 1
        self.count_reward += reward
        self.learns[self.num_of_learn] =( self.turn,self.count_reward )

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print "use turn is:",self.turn, ",count reward is now",self.count_reward

class Qlearner():
    """docstring for ."""
    def __init__(self, alpha=.5, gamma=.2 ,epsilon=.1):
        self.q = {}
        self.actions = [None, 'forward', 'left', 'right']
        self.alpha = alpha # learning rate
        self.gamma = gamma # memory / discount factor of max Q(s',a')
        self.epsilon = epsilon # probability of doing random move

    def select_action(self,state):
        q={}
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            for action in self.actions:
                q[action] = self.get_q(state, action)

            max_q = max(q.items(), key=lambda x: x[1])
            action = max_q[0]
        # print '**********2',q
        return action


    def learn_q(self, state, action, reward, value):
        q = self.q.get((state, action), None)

        if q == None:
            q = reward
        else:
            q = (1-self.alpha)*q + self.alpha * (value - q)

        self.set_q(state, action, q) #update table

    def learn(self, state, new_state, action, reward):
        q = [self.get_q(new_state, a) for a in self.actions]
        next_reward = max(q)

        self.learn_q(state, action, reward, reward - self.gamma * next_reward)

    def get_q(self, state, action):
        return self.q.get((state, action), .0)

    def set_q(self, state, action, q):
        self.q[(state, action)] = q


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print '*****',a.learns
    print '***** score is *****',a_learns_score(a.learns)

def a_learns_score(a_dict):
    score = 0
    for i in a_dict[-100:]:
        score += a_dict[i][1]/a_dict[i][0]
    return score

def show_me_the_best():
    step = .1
    alpha_list = arange(step,1+step,step)

    step = .05
    gamma_list = arange(step,.5+step,step)

    step = .05
    epsilon_list = arange(step,.5+step,step)

    find_para(alpha_list,gamma_list,epsilon_list)

def run_for_choose(alpha=.5, gamma=.2,epsilon=.1):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent,alpha=.5, gamma=.2,epsilon=.1)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0., display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    return a_learns_score(a.learns)

def find_para(alpha_list,gamma_list,epsilon_list):
    best_param_a = .5
    best_param_b = .2
    best_param_c = .1

    best_param_a,best_param_b,best_param_c,best_score = find_a_para(best_param_a,best_param_b,best_param_c,alpha_list,gamma_list,epsilon_list)
    best_param_a,best_param_b,best_param_c,best_score = find_a_para(best_param_a,best_param_b,best_param_c,alpha_list,gamma_list,epsilon_list)


    print 'best alpha is:',best_param_a,';it score is:',best_score
    print 'best gamma is:',best_param_b,';it score is:',best_score
    print 'best epsilon is:',best_param_c,';it score is:',best_score

def find_a_para(da,db,dc,list_a,list_b,list_c):
    best_a = da
    best_b = db
    best_c = dc

    best_score = 0
    best_score_now = 0

    for a in list_a:
        score_now = ( run_for_choose(alpha=a, gamma=best_b,epsilon=best_c)+run_for_choose(alpha=a, gamma=best_b,epsilon=best_c)+run_for_choose(alpha=a, gamma=best_b,epsilon=best_c))/3
        if score_now > best_score_now:
            best_score_now = score_now
            best_a = a
    best_score = best_score_now
    best_score_now = 0

    for b in list_b:
        score_now = ( run_for_choose(alpha=best_a, gamma=b,epsilon=best_c)+run_for_choose(alpha=best_a, gamma=b,epsilon=best_c)+run_for_choose(alpha=best_a, gamma=b,epsilon=best_c))/3
        if score_now > best_score_now:
            best_score_now = score_now
            best_b = b
    best_score = best_score_now
    best_score_now = 0

    for c in list_c:
        score_now = ( run_for_choose(alpha=best_a, gamma=best_b,epsilon=c) +run_for_choose(alpha=best_a, gamma=best_b,epsilon=c) +run_for_choose(alpha=best_a, gamma=best_b,epsilon=c))/3
        if score_now > best_score_now:
            best_score_now = score_now
            best_c = c
    best_score = best_score_now

    return (best_a,best_b,best_c,best_score)


if __name__ == '__main__':
    run()
    # show_me_the_best()
