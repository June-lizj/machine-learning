import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============Part1:Basic Function==============
# print('Running warmUpExercise...')
# print('5*5 identity matrix is:')
# A = np.eye(5)
# print(A)


# ==============Part2:Plotting==============
# def plotData(x,y):
#     plt.plot(x,y,'rx',ms=10)
#     plt.xlabel('population of city in 10,000s')
#     plt.ylabel('profit in $10,000s')
#     plt.show()
#
# print('plotting data....')
data = np.loadtxt('ex1data1.txt',delimiter=',')
X = data[:,0]
Y = data[:,1]
m = np.size(Y,0)

# plotData(X,Y)


# ==============Part3:Gradient descent==============
def computeCost(x,y,theta):
    cost = (x.dot(theta) - y).dot(x.dot(theta) - y)/(2*m)
    return cost

def gradientDescent(x,y,theta,alpha,num_iter):
    print(x.T)
    j_history = np.zeros(num_iter)

    for i in range(num_iter):
        deltaJ = x.T.dot(x.dot(theta) - y)/m
        theta = theta - alpha*deltaJ
        j_history[i] = computeCost(x,y,theta)
    return theta,j_history

print('running gradient descent.....')
X = np.vstack((np.ones(m),X)).T
theta = np.zeros(2)

iterations = 1500
alpha = 0.01

J = computeCost(X,Y,theta)

theta,j_history = gradientDescent(X,Y,theta,alpha,iterations)
print('theta found by gradient descent is',theta)

predict1 = np.array([1,3.5]).dot(theta)
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = np.array([1,7.0]).dot(theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)

# plt.plot(X[:,1],Y,'rx',ms=10,label='training data')
# plt.plot(X[:,1],X.dot(theta),'-',label='linear regression')
# plt.xlabel('Population of City in 10,000')
# plt.ylabel('Profit in $10,000')
# plt.legend(loc='upper right')
# plt.show()


# # ==============Part4:Visualizing J(theta_0,theta_1)==============
# print('Visualizing J(theta_0,theta_1))')
# theta0_vals = np.linspace(-10,10,100)
# theta1_vals = np.linspace(-1,4,100)
#
# J_vals = np.zeros((np.size(theta0_vals,0),np.size(theta1_vals,0)))
#
# for i in range(np.size(theta0_vals,0)):
#     for j in range(np.size(theta1_vals,0)):
#         t = np.array([theta0_vals[i],theta1_vals[j]])
#         J_vals[i,j] = computeCost(X,Y,t)
#
# # 绘制三维图像
# theta0_vals,theta1_vals = plt.meshgrid(theta0_vals,theta1_vals)
# fig = plt.figure()
# ax = plt.gca(projection='3d')
# ax.plot_surface(theta0_vals,theta1_vals,J_vals.T)
# ax.set_xlabel(r'$\theta$0')
# ax.set_ylabel(r'$\theta$1')
#
# # 绘制等高线图
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contour(theta0_vals,theta1_vals,J_vals.T,np.logspace(-2,3,20))
# ax2.plot(theta[0],theta[1],'rx',ms=10,lw=2)
# ax2.set_xlabel(r'$\theta$0')
# ax2.set_ylabel(r'$\theta$1')
# plt.show()