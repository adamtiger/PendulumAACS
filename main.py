from q_learn import wrapper_parallel_execution
import math
import graphs

init_params = [150, 'files', ((math.pi/6.0, 2.0*math.pi - math.pi/6.0), (-1.0, 1.0)), [2.0*math.pi - math.pi/3.0, 1.0]]
#init_params = [5000, 'files', ((math.pi-0.3, math.pi + 0.3), (-1.0, 1.0)), [2.0*math.pi - math.pi/3.0, 1.0]]
#init_params = [50000, 'files', ((math.pi-0.2, math.pi + 0.2), (-0.1, 0.1)), [math.pi + math.pi/3.0, 0.0]]

wrapper_parallel_execution(init_params)

#graphs.create_video(init_params)