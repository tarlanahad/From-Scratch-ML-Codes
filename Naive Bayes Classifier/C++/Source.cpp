#include<iostream>
#include<vector>
#include<algorithm>
#include<math.h>
#include<string>
#include<fstream>
#include <sstream>

using namespace std;

#define M_PI 3.14159265358979323846

class Utils {

public:
	static vector<double> meanAxis0(vector<vector<double>> X) {
		vector<double> m_vec;

		for (int j = 0; j < X[0].size(); j++)
		{
			double s = 0;
			for (int i = 0; i < X.size(); i++)
				s += X[i][j];
			m_vec.push_back(s / X.size());
		}
		return m_vec;
	}

	static vector<double> stdAxis0(vector<vector<double>> X) {
		
		vector<double> mu_vec = meanAxis0(X);

		vector<double> std_vec;
	
		for (int j = 0; j < X[0].size(); j++)
		{
			double s = 0;
			for (int i = 0; i < X.size(); i++)
				s += pow(X[i][j] - mu_vec[j],2);
			std_vec.push_back(sqrt(s / X.size()));
		}

		return std_vec;

	}

	static vector<vector<double>> read_record(string path)
	{
		vector<vector<double>> data;
		ifstream file(path);
		string str;
		while (getline(file, str))
		{
			stringstream ss(str);

			vector<double> row;

			while (ss.good())
			{
				string substr;
				getline(ss, substr, ',');
				row.push_back(stod(substr));
			}
			data.push_back(row);
		}

		return data;
	}

	static void printMat(vector<vector<double>> A) {
		for (vector<double> row : A) {
			printVec(row);
			cout << endl;
		}
	}

	static void printVec(vector<double> X) {
		for (double x : X)
			cout << x << " ";
	}

	static vector<double> unique(vector<double> y) {
		vector<double> unique;
		for (double a : y)
			if (count(unique.begin(), unique.end(), a) == 0)
				unique.push_back(a);
		return unique;
	}

};

class NaiveBayes {

public:

	NaiveBayes(vector<vector<double>> X, vector<double> y) {
		this->X = X;
		this->y = y;
	}

	void fit() {

		classes = Utils::unique(this->y);

		for (double c : classes) {

			vector<vector<double>> x_c;
			for (int i = 0; i < y.size(); i++)
				if (c == y[i])
					x_c.push_back(X[i]);

			means.push_back(Utils::meanAxis0(x_c));
			stds.push_back(Utils::stdAxis0(x_c));

			priors.push_back(double(x_c.size()) / X.size());
		
		}

	}

	vector<double> predict(vector<vector<double>> X) {
		vector<double> y_pred;
		for (vector<double> x : X)
			y_pred.push_back(__predict__(x));
		return y_pred;
	}


private:
	vector<double> classes,priors;
	vector<vector<double>> means, stds;
	vector<vector<double>> X;
	vector<double> y;

	

	vector<double> pdf(int idx, vector<double> x) {
			
		vector<double> mu = means[idx];
		vector<double> std = stds[idx];

		vector<double> gaussian_pdf;

		for (int i = 0; i < mu.size(); i++) {
			double numerator = exp(-pow(x[i] - mu[i], 2) / (2 * pow(std[i],2)));
			double denominator = sqrt(2 * M_PI * pow(std[i],2));
			gaussian_pdf.push_back(numerator / denominator);
		}

		return gaussian_pdf;
	}

	double __predict__(vector<double> x) {
		
		double max_posterior = -99999, max_idx = -1;

		for (int idx = 0; idx < classes.size(); idx++)
		{
			double c = classes[idx];
			double prior = priors[idx];
			
			double posterior = 0;
			vector<double> pdfs = pdf(idx, x);
			for (double p : pdfs) {
				posterior += log10(p);
			}

			posterior = posterior + log10(prior);
			
			if (posterior > max_posterior) {
				max_posterior = posterior;
				max_idx = idx;
			}

		}

		return classes[max_idx];
	}

};




int main() {
	
	vector<vector<double>> A = { {2,1,3},{3,4,5} };


	
	vector<vector<double>> data = Utils::read_record("Files/data_set.csv");

	vector<vector<double>> train_x;
	vector<vector<double>> test_x;

	vector<double>train_y;
	vector<double>test_y;

	int train_count = int(0.8 * data.size());
	int test_count = data.size() - train_count;


	for (int i = 0; i < train_count; i++)
	{
		vector<double> row;
		for (int j = 0; j < data[0].size(); j++)
		{
			if (j == 0)
				train_y.push_back(data[i][j]);
			else
				row.push_back(data[i][j]);
		}
		train_x.push_back(row);
	}




	for (int i = train_count; i < train_count + test_count; i++)
	{
		vector<double> row;
		for (int j = 0; j < data[0].size(); j++)
		{
			if (j == 0)
				test_y.push_back(data[i][j]);
			else
				row.push_back(data[i][j]);
		}
		test_x.push_back(row);
	}

	

	NaiveBayes nb = NaiveBayes(train_x, train_y);
	nb.fit();


	vector<double> preds_y = nb.predict(test_x);


	double counter = 0;
	for (int i = 0; i < preds_y.size(); i++)
		if (preds_y[i] == test_y[i])
			counter += 1;
	cout << "Accuracy: " << counter / preds_y.size() << endl;
		
	

	

	system("PAUSE");
	return 0;
}
