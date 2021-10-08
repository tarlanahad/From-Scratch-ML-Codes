/*


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


*/


#include<iostream>
#include<vector>
#define _USE_MATH_DEFINES
#include<math.h>

using namespace std;

double errorf(vector<double> current_pos) {

	double x = current_pos[0];
	double y = current_pos[1];

	return (x * x - 10 * cos(2 * M_PI * x)) +
		(y * y - 10 * cos(2 * M_PI * y)) + 20;
}

vector<double> grad_error(vector<double> current_pos) {
	
	double x = current_pos[0];
	double y = current_pos[1];

	return vector<double> {2 * x + 10 * 2 * M_PI * x * sin(2 * M_PI * x),
		2 * y + 10 * 2 * M_PI * y * sin(2 * M_PI * y)};

}

class Utils {

public:
	static double fRand(double fMin, double fMax)
	{
		double f = (double)rand() / RAND_MAX;
		return fMin + f * (fMax - fMin);
	}

	static vector<double> uniform_rand_vec(int dim, double fMin, double fMax) {
		vector<double> a;
		for (int i = 0; i < dim; i++)
			a.push_back(fRand(fMin, fMax));
		return a;
	}

	static vector<double> add2v(vector<double> a, vector<double> b){
		vector<double> c;
		for (int i = 0; i < a.size(); i++)
			c.push_back(a[i] + b[i]);
		return c;
	}

	static vector<double> mulScal2V(vector<double> a, double scal) {
		vector<double> b;
		for (double i : a)
			b.push_back(scal * i);
		return b;
	}
};

class Particle {

public:
	vector<double> position;
	vector<double> velocity;
	vector<double> best_part_pos;
	double error, best_part_error;


	Particle(int dim, double minx, double maxx) {
		position = Utils::uniform_rand_vec(dim, minx, maxx);
		velocity = Utils::uniform_rand_vec(dim, minx, maxx);
		best_part_pos = position;

		error = errorf(position);
		best_part_error = error;
	}

	void set_pos(vector<double> pos) {
		position = pos;
		error = errorf(pos);
		if (error < best_part_error) {
			best_part_error = error;
			best_part_pos = pos;
		}
	}


	/*
       
	*/
};

class PSO {

public:
	double w = 0.729,/// FINE - 
		c1 = 1.49445,///  - TUNE
		c2 = 1.49445,/// THESE
		lr = 0.01,///    VARIABLES


		best_swarm_error = 1e20;
	int numOfEpochs;

	vector<double> best_swarm_position;
	vector<Particle> swarm_list;

	PSO(int dims, int numOfBoids, int numOfEpochs) {
		for (int i = 0; i < numOfBoids; i++)
			swarm_list.push_back(Particle(dims, -500, 500));
		this->numOfEpochs = numOfEpochs;
		this->best_swarm_position = Utils::uniform_rand_vec(2, -500, 500);
	}

	void optimize() {
		for (int  i = 0; i < numOfEpochs; i++)
		{
			for (int j = 0; j < swarm_list.size(); j++)
			{
				Particle current_particle = swarm_list[j];

				vector<double> Vcurr = grad_error(current_particle.position);

				vector<double> n_curr_pos = Utils::mulScal2V(current_particle.position, -1);

				
				vector<double> deltaV = Utils::mulScal2V(Vcurr, w);
				
				deltaV = Utils::add2v(deltaV, Utils::mulScal2V(Utils::add2v(current_particle.best_part_pos, n_curr_pos), c1));

				deltaV = Utils::add2v(deltaV, Utils::mulScal2V(Utils::add2v(best_swarm_position, n_curr_pos), c2));

				vector<double> new_position = Utils::add2v(current_particle.position, Utils::mulScal2V(deltaV, -1 * lr));

				swarm_list[j].set_pos(new_position);

				if (swarm_list[j].error < best_swarm_error) {
					best_swarm_position = swarm_list[j].position;
					best_swarm_error = swarm_list[j].error;
				}


			}

			cout << "Epoch: " << i <<
				" | Best position: [ " << best_swarm_position[0] << ", " << best_swarm_position[1] <<
				" ] | Best known error: " << best_swarm_error << endl;

		}
	}


};

void main() {

	PSO pso = PSO(2, 1000, 200);
	pso.optimize();

	system("PAUSE");
}