import os;
import sys;
import numpy as np;
import datetime as dt;
from numpy import linalg as LA;


class DBReader:
	def	__init__(self,filetoread):
		self.file_id = filetoread;
	# Read file and assign data and class values
	def	readFile(self):
		# stores each instance of the database
		self.X = np.array([]);
		self.Y = [];
		for	line	in	self.file_id:
			# irisslwc.txt #0.12 0.32,iris-Other
			tuplex= line.split("\t");			
			tuplex1= tuplex[1].split(",");
			if self.X.size:
				t = [];
				t.append(float(tuplex[0]));
				t.append(float(tuplex1[0]));
				self.X = np.vstack([self.X,t]);
			else:
				t = [];
				t.append(float(tuplex[0]));
				t.append(float(tuplex1[0]));
				self.X = np.array(t);
			if tuplex1[1] == 'Iris-Other\n':
				self.Y.append(1);
			else:
				self.Y.append(-1);
		return (self.X,self.Y);

	def	find_quad_x(self):
		size = self.X.shape;
		Quad_x = np.array([]);
		root_2 = 2 ** 0.5;
		for i in self.X:
			temp = np.array([]);
			temp = np.append(temp,i[0] **2);		# x1^2
			temp = np.append(temp,i[1] **2);		# x2^2
			temp = np.append(temp,2 * i[0]);		# 2 * x1
			temp = np.append(temp,2 * i[1]);		# 2 * x2
			temp = np.append(temp, root_2 * i[0] * i[1]);	# root2 * x1 * x2
			temp = np.append(temp, 2)			# 1 for non-homogeneous +1 for bias
			if Quad_x.any():
				Quad_x = np.vstack([Quad_x,temp]);
			else:
				Quad_x = np.array(temp);
		return Quad_x;

	def	map_X_higher(self):
		size = self.X.shape;
		new_X = np.array([]);
		for i in self.X:
			i = np.append(i,1);
			if new_X.any():
				new_X = np.vstack([new_X,i]);
			else:
				new_X = np.array(i);
		self.X = new_X;
		self.size = self.X.shape;
		return self.X;

	def	cal_linear_kernel(self):
		self.lin_K = np.zeros([self.size[0],self.size[0]]);
		for i in range(0,self.size[0]):
			for j in range(i,self.size[0]):
				self.lin_K[i][j] = np.dot(self.X[i].reshape(1,self.size[1]),self.X[j].reshape(self.size[1],1));
				self.lin_K[j][i] = self.lin_K[i][j];
		return	self.lin_K;		

	def	cal_inhmogeneous_kernel(self):		
		Qud_K = np.zeros([self.size[0],self.size[0]]);
		for i in range(0,self.size[0]):
			for j in range(i,self.size[0]):
				Qud_K[i][j] = pow(self.lin_K[i][j]+1,2);
				Qud_K[j][i] = Qud_K[i][j];
		return Qud_K;


#End of DBReader class
#############################################################################################################
def	find_gradient(k,alpha_new,Y,K):
	total = 0;
	for i in range(0,alpha_new.size):
		total = total + alpha_new[i]*Y[i]*K[k][i];
	return (1 - Y[k]*total);

#############################################################################################################
def	SVM_Grad_asc(X,Y,K,C,epsilon):
	#
	size = K.shape;
	eta = np.zeros(size[0]);
	for i in range(0,size[0]):
		eta[i] = 1.0 / K[i][i];
	# 
	alpha_new = np.zeros(size[0]);
	i =0;
	while True:
		old_alpha = np.copy(alpha_new);
		for k in range(0,alpha_new.size):
			second = find_gradient(k,alpha_new,Y,K);
			alpha_new[k] = alpha_new[k] + (eta[k] * second);
			if alpha_new[k] < 0:
				alpha_new[k] = 0;
			if alpha_new[k] > C:
				alpha_new[k] = C;
		diff = np.linalg.norm((alpha_new - old_alpha), ord=2);
		i = i+1;
		if diff <= epsilon:
			#print old_alpha;
			#print alpha_new;
			#print i;
			#print diff;
			break;
	return alpha_new;

def	find_W(alpha, X, Y):
	total = np.zeros((X.shape[1])-1);
	for i in range(0,alpha.size-1):
		new_x = X[i][0:(X.shape[1])-1]
		total = total + alpha[i]*Y[i]*new_x;
	return total;

def	find_avg_b(alpha,W,X,Y,C):
	total = 0.0;
	num = 0.0;
	for i in range(0,alpha.size-1):
		if alpha[i] < C and alpha[i] > 0:
			new_x = X[i][0:(X.shape[1])-1];
			total = total + (Y[i] - np.dot(W.reshape(1,W.size),new_x.reshape(W.size,1)));
			num = num +1 ;
	return total/num;

 

#############################################################################################################
def 	main():
	if	len(sys.argv) < 2:
		print 'Please give me the filename'+os.linesep;
		sys.exit(1);
	try:	
		#file name
		f_name = sys.argv[1];
		fileToRead=open(f_name);
		#fileToRead = open("data.txt");	
	except IOError,IndexError:
		print	'Bad file name'+os.linesep;
		sys.exit(1);
	#Given Values
	C = 10;
	epsilon = 0.0001; 

	try:
#		f_linear = open('linear-kernel.txt','w');
		f_quad = open('Non-Homogeneous-quadratic-kernel.txt','w');
		f_equ = open('Non-Homogenous-equation.txt','w');
	except IOError,IndexError:
		print	'output file can not be opened'+os.linesep;
		sys.exit(1);

	reader = DBReader(fileToRead);
	(X,Y) = reader.readFile();
	Quad_X = reader.find_quad_x();
	X = reader.map_X_higher();

	Linear_K = reader.cal_linear_kernel();
#	Homog_Qud_K = reader.cal_hmogeneous_kernel();	
	InHo_Qud_K = reader.cal_inhmogeneous_kernel();

	alpha1 = SVM_Grad_asc(Quad_X,Y,InHo_Qud_K,C,epsilon);
	for a in alpha1:
		f_quad.write(repr(a) +'\n');
	num_sup1 = (alpha1 != 0.0).sum();
	f_quad.write('Number of support vectors: '+ repr(num_sup1) +'\n');

#	print alpha;
	print alpha1;

	W1 = find_W(alpha1,Quad_X,Y);					# weights are multiplied by root_2 values in Quad_X
	b1 = find_avg_b(alpha1,W1,Quad_X,Y,C);
	f_equ.write('Non-Homogeneous-Quadratic Kernel Equation:'+'\n'+repr(W1[0])+' * X1^2');
	if W1[1] >= 0:
		f_equ.write(' + '+repr(W1[1])+' * X2^2');
	else:
		f_equ.write(' '+repr(W1[1])+' * X2^2');
	if W1[2] >= 0:
		f_equ.write(' + '+repr(W1[2] * 2)+' * X1');
	else:
		f_equ.write(' '+repr(W1[2] * 2)+' * X1');
	if W1[3] >= 0:
		f_equ.write(' + '+repr(W1[3] * 2)+' *X2');
	else:
		f_equ.write(' '+repr(W1[3] * 2)+' *X2');
	if W1[4] >= 0:
		f_equ.write(' + '+repr(W1[4] * (2 ** 0.5))+' * X1*X2');
	else:
		f_equ.write(' '+repr(W1[4] * (2 ** 0.5))+' * X1*X2');
	if b1[0][0] >= 0:
		f_equ.write(' + '+repr(b1[0][0])+' = 0\n');
	else:
		f_equ.write(' '+repr(b1[0][0])+' = 0\n');
	
#	print W;
	print W1;
#	print b;
	print b1;

#End of main function
###################################################################################################################
if	__name__== "__main__":
	main();
