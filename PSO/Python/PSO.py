"""''

Copyright © 2020 Tarlan Ahadli

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

''"""

import numpy as np
import matplotlib.pyplot as plt


def error(current_pos):
    x = current_pos[0]
    y = current_pos[1]

    return (x ** 2 - 10 * np.cos(2 * np.pi * x)) + \
           (y ** 2 - 10 * np.cos(2 * np.pi * y)) + 20


def grad_error(current_pos):
    x = current_pos[0]
    y = current_pos[1]

    return np.array(
        [2 * x + 10 * 2 * np.pi * x * np.sin(2 * np.pi * x),
         2 * y + 10 * 2 * np.pi * y * np.sin(2 * np.pi * y)])


class PSO:
    class Particle:

        def __init__(self, dim, minx, maxx):
            self.position = np.random.uniform(low=minx, high=maxx, size=dim)
            self.velocity = np.random.uniform(low=minx, high=maxx, size=dim)
            self.best_part_pos = self.position.copy()

            self.error = error(self.position)
            self.best_part_err = self.error.copy()

        def setPos(self, pos):
            self.position = pos
            self.error = error(pos)
            if self.error < self.best_part_err:
                self.best_part_err = self.error
                self.best_part_pos = pos

    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    lr = 0.01

    def __init__(self, dims, numOfBoids, numOfEpochs):
        self.swarm_list = [self.Particle(dims, -500, 500) for i in range(numOfBoids)]
        self.numOfEpochs = numOfEpochs

        self.best_swarm_position = np.random.uniform(low=-500, high=500, size=dims)
        self.best_swarm_error = 1e80  # Set high value to best swarm error

    def optimize(self):
        for i in range(self.numOfEpochs):

            for j in range(len(self.swarm_list)):

                current_particle = self.swarm_list[j]  # get current particle

                Vcurr = grad_error(current_particle.position)  # calculate current velocity of the particle

                deltaV = self.w * Vcurr \
                         + self.c1 * (current_particle.best_part_pos - current_particle.position) \
                         + self.c2 * (self.best_swarm_position - current_particle.position)  # calculate delta V

                new_position = self.swarm_list[j].position - self.lr * deltaV  # calculate the new position

                self.swarm_list[j].setPos(new_position)  # update the position of particle

                if error(new_position) < self.best_swarm_error:  # check the position if it's best for swarm
                    self.best_swarm_position = new_position
                    self.best_swarm_error = error(new_position)

            print('Epoch: {0} | Best position: [{1},{2}] | Best known error: {3}'.format(i,
                                                                                         self.best_swarm_position[0],
                                                                                         self.best_swarm_position[1],
                                                                                         self.best_swarm_error))


pso = PSO(dims=2, numOfBoids=30, numOfEpochs=500)
pso.optimize()
