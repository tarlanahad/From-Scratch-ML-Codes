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
#include<cmath>
#include <stdlib.h>     
#include <time.h> 
#include <algorithm>       



using namespace std;


class kmeans {

public:
	vector<vector<double>> centroids;
	vector<vector<vector<double>>> cluster_list;

	kmeans(vector<vector<double>> X, int k, int max_iters) {
		this->X = X;
		this->k = k;
		this->max_iters = max_iters;
	}


	kmeans fit() {


		for (int i = 0; i < k; i++) {
			vector<double> cetroid;
			for (int j = 0; j < X[0].size(); j++)
				cetroid.push_back((double)(rand() % 100));
			centroids.push_back(cetroid);
		}

		bool converged = false;
		int current_iter = 0;

		while (!converged && current_iter < max_iters) {

			cluster_list.clear();
			for (int i = 0; i < centroids.size(); i++) {
				vector<vector<double>> a;
				cluster_list.push_back(a);
			}

			for (vector<double> x : X) {
				vector<double> distances_list;
				for (vector<double> c : centroids)
					distances_list.push_back(getDistance(x, c));
				int minIdex = min_element(distances_list.begin(), distances_list.end()) - distances_list.begin();
				cluster_list[minIdex].push_back(x);
			}

			// Remove empty clusters
			vector<int> emptyIndexes;
			for (int i = 0; i < cluster_list.size(); i++)
				if (cluster_list[i].empty())
					emptyIndexes.push_back(i);

			for (int i = 0; i < emptyIndexes.size(); i++)
				cluster_list.erase(cluster_list.begin() + (emptyIndexes[i] - i));

			vector<vector<double>> prev_centroids;
			for (vector<double> c : centroids)
				prev_centroids.push_back(c);

			centroids.clear();

			for (int i = 0; i < cluster_list.size(); i++)
				centroids.push_back(get_column_mean(cluster_list[i]));

			double pattern = abs(getMatrixSum(centroids) - getMatrixSum(prev_centroids));

			cout << "K-MEANS: " << (int)pattern << endl;

			converged = pattern == 0;
			current_iter++;

		}

		return *this;
	}


private:

	vector<vector<double>> X;
	int k;
	int max_iters;

	vector<double> get_column_mean(vector<vector<double>> cluster) {

		vector<double> n_centroid;

		for (int m = 0; m < cluster[0].size(); m++) {
			double sum = 0;
			for (int n = 0; n < cluster.size(); n++)
				sum += cluster[n][m];
			n_centroid.push_back(sum / cluster.size());
		}

		return n_centroid;
	}

	double getDistance(vector<double> x1, vector<double> x2) {
		double sum = 0;
		for (int i = 0; i < x1.size(); i++)
			sum += pow(x1[i] - x2[i], 2);
		return sqrt(sum);
	}

   double getMatrixSum(vector<vector<double>> c) {
		double sum = 0;
		for (int i = 0; i < c.size(); i++)
			for (int j = 0; j < c[0].size(); j++)
				sum += c[i][j];
		return sum;
	}


};

void print2Dvector(vector<vector<double>> vec) {
	for (vector<double> row : vec) {
		for (double a : row)
			cout << a << " ";
		cout << endl;
	}
}


void main()
{
	vector<vector<double>> 
		a{{2,4,1},
		{-1,1,-1},
		{1,4,0}};

	kmeans classifier = kmeans(a, 2, 100).fit();

	cout << "------Centroids------" << endl;
	print2Dvector(classifier.centroids);



	system("PAUSE");
}
