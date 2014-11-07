import os;
import sys;
import numpy as np;
import datetime as dt;
from numpy import linalg as LA;
import ast;
import random as rnd;


def tryeval(val):
  try:
    val = ast.literal_eval(val)
  except ValueError:
    pass
  return val
############################################################################################################
def	get_nearest_mean(curr,means):
	mean_id = -1;
	min_val = sys.float_info.max;
	min_id = 0;
	for i in means:
		mean_id = mean_id + 1;
		diff = np.linalg.norm((curr - i), ord=2);
		sqr = diff  * diff;
		if sqr < min_val:
			min_val = sqr;
			min_id = mean_id;
	return min_id;
############################################################################################################
def 	Kmeans(X,K,means,epsilon):
	start_time = dt.datetime.now();	
	t = 0;
	s = X.shape;
	data_cluster_ids = np.zeros(s[0]);
	clusters = [set() for index in xrange(K)];
	while True:
		clusters = [set() for index in xrange(K)];
		t = t + 1;
		data_id = 0;
		# assign cluster id to data
		for x_j in X:
			data_id = data_id + 1;
			i = get_nearest_mean(x_j,means);
			data_cluster_ids[data_id-1] = i;
			clusters[i].add(data_id);

		old_means = np.copy(means);

		# get sum of difference of all means
		sum_diff = 0;

		# find mean for new clusters
		for i in range(K):
			l = len(clusters[i]);
			sum_v = 0;
			for t in clusters[i]:
				sum_v = sum_v + X[t-1];
			means[i] = sum_v / l;
			
			# calculate difference
			diff = np.linalg.norm((means[i] - old_means[i]), ord=2);
			sum_diff = sum_diff + diff;
		
		if sum_diff <= epsilon:
			break;	
	end_time = dt.datetime.now();
	time = end_time - start_time;
	return (data_cluster_ids,means,clusters,time,t);
##############################################################################################################
def	find_SSE(X,clusters,means):
	S = 0;
	for i in range(len(clusters)):
		sum_v = 0;
		for j in clusters[i]:
			diff = np.linalg.norm((X[j-1] - means[i]), ord=2);
			sqr = diff  * diff;
			sum_v = sum_v + sqr;
		S = S + sum_v;
	return S;
##############################################################################################################
class DBReader:
	def	__init__(self,filetoread):
		self.file_id = filetoread;
	# Read file and assign data and class values
	def	readFile(self):
		self.data_count = 0;
		self.dimentions = 0;
		# stores each instance of the database
		self.X = np.array([]);
		self.Y = [];
		for	line	in	self.file_id:		
			self.data_count = self.data_count + 1;
			tuplex= line.split(",");
			if self.X.size:
				last = tuplex[self.dimentions -1].split("\n");
				tuplex[self.dimentions -1] = last[0];
				t = [tryeval(x) for x in tuplex]
				self.X = np.vstack([self.X,t]);
			else:
				self.dimentions = len(tuplex);
				last = tuplex[self.dimentions -1].split("\n");
				tuplex[self.dimentions -1] = last[0];
				t = [tryeval(x) for x in tuplex]
				self.X = np.array(t);
		return (self.X,self.data_count,self.dimentions);
#-----------------------------------------------------------------------------------
	def	get_random_means(self,K):
		rnd.seed();					#takes current system time as seed
		return rnd.sample(range(self.data_count),K);
#-----------------------------------------------------------------------------------
	def	get_means_from_file(self,fileToRead):
		return [tryeval(x) for x in fileToRead];
#-----------------------------------------------------------------------------------
	def	get_mean_values(self,ids):
		means = np.array([]);
		for i in ids:
			t = self.X[i-1];						# line no starts with 1 & array with 0
			if means.size:
				means = np.vstack([means,t]);
			else:
				means = np.array(t);
		return np.array(means);
#############################################################################################################

def 	main():
	if	len(sys.argv) < 3:
		print 'Please give me the Data Filename + # of means(k) '+os.linesep;
		sys.exit(1);
	else:
		try:	
			#file name
			f_name = sys.argv[1];
			fileToRead=open(f_name);
			#fileToRead = open("data.txt");
		except IOError,IndexError:
			print	'Bad file name'+os.linesep;
			sys.exit(1);

	no_means_K = int(sys.argv[2]);
	reader = DBReader(fileToRead);			
	(X,data_count,dimentions) = reader.readFile();

	if	len(sys.argv) == 4:
		try:	
			# means file name
			means_f_name = sys.argv[3];
			means_fileToRead=open(means_f_name);
			#fileToRead = open("data.txt");	
		except IOError,IndexError:
			print	'Bad file name'+os.linesep;
			sys.exit(1);
		mean_ids = reader.get_means_from_file(means_fileToRead);
	else:
		mean_ids = reader.get_random_means(no_means_K);

	means = reader.get_mean_values(mean_ids);
#checking
	print means;

	epsilon = 0.001;
	
	s = X.shape;
	print 'Number of data points = '+str(s[0]);
	print 'Dimentions of the data = '+str(dimentions);
	print 'Number of clusters = '+str(no_means_K);

	(data_clusters,final_means,clusters,time,t) = Kmeans(X,no_means_K,means,epsilon);

	print 'Computation time = '+ str(time.total_seconds()) + ' Seconds';
	print 'Number of Iterations = '+str(t);
	
	SSE = find_SSE(X,clusters,final_means);

	print 'Sum of Squere Error = '+str(SSE);

	for i in range(no_means_K):
		print '********************************************';
		print 'Size of Cluster-'+str(i+1)+' = '+str(len(clusters[i]));
		print 'Data ID in Cluster-'+str(i+1)+' = '+str(clusters[i]);

	# for cheking
#	print 'Means: '+str(final_means);

	#Check_purity(data_clusters,s[0]);

#End of main function
###################################################################################################################
if	__name__== "__main__":
	main();
