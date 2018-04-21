import gnrlz_q as gz
from q_learn import wrapper_parallel_execution
import argparse
import logger
import graphs
import math

#init_params = [150, 'files', ((math.pi/6.0, 2.0*math.pi - math.pi/6.0), (-1.0, 1.0)), [2.0*math.pi - math.pi/3.0, 1.0]]
#init_params = [5000, 'files', ((math.pi-0.3, math.pi + 0.3), (-1.0, 1.0)), [2.0*math.pi - math.pi/3.0, 1.0]]
#init_params = [50000, 'files', ((math.pi-0.2, math.pi + 0.2), (-0.1, 0.1)), [math.pi + math.pi/3.0, 0.0]]

#wrapper_parallel_execution(init_params)

#graphs.create_video(init_params)

parser = argparse.ArgumentParser(description='Generalization over Pendulum')

parser.add_argument('--mode', default=0, type=int, metavar='N',
        help='The mode of the run. 0: daemon training, 1: regression')

args = parser.parse_args()

def assemble_dataset():
    '''
    the goals are arranged on the circle
    starts at 60 degree then repeated with
    15 degrees frequency
    '''

    init_params = []

    for num in range(17):
        goal = [math.pi/3.0 + num * math.pi/12.0, 0.0]
        if goal[0] < math.pi:
            start = goal[0] - math.pi/6.0
            end = 2.0 * math.pi - start
        else:
            end = goal[0] + math.pi / 6.0
            start = 2.0 * math.pi - end

        init_params.append(
            [150, 'files/params' + str(num), ((start, end), (-1.0, 1.0)), goal]
        )

    return init_params


def run_training_batch():

    params = assemble_dataset()

    for p in params:
        wrapper_parallel_execution(p)


# first do the training
if args.mode == 0:
    run_training_batch()
elif args.mode == 1:
    base_folder = 'files/params'
    reg = gz.Regression([(base_folder + str(x), x) for x in range(1)])
    h = reg.regression(32, 2)
    log = logger.Logger('files')
    log.log(logger.Mode.STD_LOG, str(h.history))
    print('Finish.')
else:
    print('Unknown mode!')



