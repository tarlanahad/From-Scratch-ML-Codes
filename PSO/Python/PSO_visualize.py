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


X = np.linspace(-500, 500, 2000)
Y = np.linspace(-500, 500, 2000)
X, Y = np.meshgrid(X, Y)

Z = error([X, Y])

fig, ax = plt.subplots(nrows=1, ncols=2)

fig.set_size_inches(20, 10)

ax[0].contour(X, Y, Z, 150)
ax[0].set_xlim([-500, 500])
ax[0].set_ylim([-500, 500])
plt.show()

x_arr = []
y_arr = []


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
        self.best_swarm_error = 1e80

    def getAllPartPosList(self):
        return np.array([part.position for part in self.swarm_list])

    def optimize(self):
        for i in range(self.numOfEpochs):

            current_positions = self.getAllPartPosList()

            path = ax[0].scatter(current_positions[:, 0], current_positions[:, 1], c='r')

            for j in range(len(self.swarm_list)):

                current_particle = self.swarm_list[j]

                Vcurrent = grad_error(current_particle.position)

                Vnext = self.w * Vcurrent \
                        + self.c1 * (current_particle.best_part_pos - current_particle.position) \
                        + self.c2 * (self.best_swarm_position - current_particle.position)

                new_position = self.swarm_list[j].position - 0.01 * Vnext

                self.swarm_list[j].setPos(new_position)

                if error(new_position) < self.best_swarm_error:
                    self.best_swarm_position = new_position
                    self.best_swarm_error = error(new_position)

            plt.figure(1)
            x_arr.append(i)
            y_arr.append(self.best_swarm_error)
            ax[1].plot(x_arr, y_arr, c='g')
            ax[1].set_title('Best error so far: {0}'.format(self.best_swarm_error))
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Error')

            plt.pause(0.0001)
            path.remove()

            print('Epoch: {0} | Best position: [{1},{2}] | Best known error: {3}'.format(i,
                                                                                         self.best_swarm_position[0],
                                                                                         self.best_swarm_position[1],
                                                                                         self.best_swarm_error))


cool = PSO(dims=2, numOfBoids=20, numOfEpochs=300)
cool.optimize()
