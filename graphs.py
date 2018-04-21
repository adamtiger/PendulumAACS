import q_learn


def create_video(init_params):
    max_ep = init_params[0]
    folder = init_params[1]
    constraints = init_params[2]
    goal = init_params[3]

    q = q_learn.Q(max_ep, folder, constraints, goal)
    q.load_from_file()

    rtn, scs = q.evaluate(video=False, show=True)
    print(str(rtn) + ' ' + str(scs))
