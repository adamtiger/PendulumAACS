import q_learn
import math
import gym
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import random
import numpy as np

def create_video(init_params):
    max_ep = init_params[0]
    folder = init_params[1]
    constraints = init_params[2]
    goal = init_params[3]

    q = q_learn.Q(max_ep, folder, constraints, goal)
    q.load_from_file()

    rtn, scs = q.evaluate(video=False, show=True)
    print(str(rtn) + ' ' + str(scs))


def create_bar_plot():
    init_params = []

    goals = []
    for num in range(17):
        goal = [math.pi / 3.0 + num * math.pi / 12.0, 0.0]
        if goal[0] < math.pi:
            start = goal[0] - math.pi / 6.0
            end = 2.0 * math.pi - start
        else:
            end = goal[0] + math.pi / 6.0
            start = 2.0 * math.pi - end

        init_params.append(
            [150, 'files/params' + str(num), ((start, end), (-1.0, 1.0)), goal]
        )

        goals.append(goal)

    # calculate successes for the generalized
    net = Sequential()
    net.add(Dense(5, input_shape=(4,)))
    net.add(Activation('relu'))
    net.add(Dense(24))
    net.add(Activation('relu'))
    net.add(Dense(2))
    net.add(Activation('softmax'))

    net.load_weights('files/tmp500_weights.h5')
    net.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    def do_eval_goal(p, net):

        constraints = p[2]
        goal = p[3]

        env = gym.make('PendulumGoal-v0')

        def __env_init_fn():
            env.reset()
            th = random.uniform(constraints[0][0], constraints[0][1])
            thdot = random.uniform(constraints[1][0], constraints[1][1])
            env.setup([np.array([th, thdot]), goal])
            return np.array([math.cos(th), math.sin(th), thdot])

        def __select_act(state):
            epsilon = 0.1
            best_act = np.argmin(net.predict(state)[0])
            dice = random.randint(1, 1000)
            if dice > epsilon * 1000:
                return best_act * 2.0 - 1.0
            else:
                idcs = [idx for idx in range(0, 2)]
                idcs.remove(best_act)
                return random.choice(idcs) * 2.0 - 1.0

        cntr = 0
        episode = 0
        done = True
        success = -1
        state = None

        while episode < 20:

            if done:
                success += 1

            if done or cntr > 2000:

                episode += 1
                state = __env_init_fn()

            c, s, dot = state[0], state[1], state[2]
            theta = math.acos(math.fabs(c))
            if c >= 0 and s >= 0:
                theta = theta
            elif c < 0 and s > 0:
                theta = math.pi - theta
            elif c < 0 and s < 0:
                theta = math.pi + theta
            else:
                theta = 2.0 * math.pi - theta

            state_r = np.array([[theta, dot, goal[0], goal[1]]])
            action = __select_act(state_r)
            state, cost, done, _ = env.step([action])
            cntr += 1

        return success

    print('Measure performance on generalized.')
    successes_gen = []
    for p in init_params:

        scs = do_eval_goal(p, net)
        successes_gen.append(scs)

    print('Measure performance on original.')
    successes = []
    for p in init_params:
        max_ep = p[0]
        folder = p[1]
        constraints = p[2]
        goal = p[3]

        q = q_learn.Q(max_ep, folder, constraints, goal)
        q.load_from_file()

        rtn, scs = q.evaluate(video=False, show=False)
        successes.append(scs)

    ind = np.arange(17)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, successes, width)
    p2 = plt.bar(ind, successes_gen, width, bottom=successes)

    plt.ylabel('NUmber of successes out of 20')
    plt.title('Successes by goals')
    gs = ('60', '75', '90', '105', '120', '135', '150',
          '165', '180', '195', '210', '225', '240', '255',
          '270', '285', '300', '315', '330', '345', '360')
    plt.xticks(ind, gs)
    plt.yticks(np.arange(0, 41, 2))
    plt.legend((p1[0], p2[0]), ('Direct', 'Generalized'))

    plt.show()

create_bar_plot()

