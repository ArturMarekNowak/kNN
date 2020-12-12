import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import time
import csv
import pprint
import random
from sklearn import neighbors


# ==============================================================================
# This is a functions which helps in creation in dictionary from parsed dataset.
# It goes through chosen features and creates a tuple from them.
# ==============================================================================
def initHelper(features, chosenFeatures):
    retTup = []

    for ft in chosenFeatures:
        retTup.append(features[ft])

    return retTup


# ========================================================================================
# This function takes as an input number of dimensions/features we want to extract from
# dataset and "mod" parameter which makes parser to take each "mod-th" point from dataset,
# for example if mod = 10 the parser will take each 10th point from dataset.
# As an output function returns a dictionary of classified points.
# ========================================================================================
def getPoints(dim=3, mod=10000):
    pointsXD = {0: [], 1: []}
    chosenFeatures = [6, 10, 14, 15, 17, 19, 20, 21, 22]
    chosenFeatures = chosenFeatures[:dim]

    # drive.mount('/content/drive/', force_remount=True)

    with open('datasets\Malware dataset - Malware dataset.csv', 'r') as f:
        data = csv.reader(f, delimiter=",")

        itr = 1  # Skip first row with collumn names
        for row in data:
            if (itr % mod) == 0:
                if row[2] == 'malware':
                    rc = initHelper(row, chosenFeatures)
                    rc = [float(item) for item in rc]
                    pointsXD[1].append(rc)
                else:
                    rc = initHelper(row, chosenFeatures)
                    rc = [float(item) for item in rc]
                    pointsXD[0].append(rc)
            itr += 1

    print("success!!!!")
    # drive.flush_and_unmount()

    return pointsXD


# Calling getPoints() function in order to extract dictionary of points and drawing random point
pointsXD = getPoints()

p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

k1 = 1
k3 = 3
k4 = 5
k5 = 7


# ==================================================================================================================
# This function takes as an input a dictionary of classified points and normalizes them to scope of 0 to 1.

# maxValues list is initialized by zeros with length equal to number of features of points in dictionary.
# Then next lists of features from dataset are compare with # values in maxValues list. If there is any bigger value,
# the maxValues list is updated.
# In the second part all features in dataset are divided by the biggest value.
# ==================================================================================================================
def normalization(pointsDict=pointsXD):
    # Simplest idea:
    maxValues = [0] * len(pointsDict[0][0])

    # Find max values in each column
    for key in pointsDict:
        for onePoint in pointsDict[key]:
            for feature in range(len(onePoint)):
                if maxValues[feature] < onePoint[feature]:
                    maxValues[feature] = onePoint[feature]

    # Normalize
    for key in pointsDict:
        for onePoint in pointsDict[key]:
            for feature in range(len(onePoint)):
                onePoint[feature] = (onePoint[feature] / maxValues[feature])
                onePoint[feature] = round(onePoint[feature], 2)

    return pointsDict


pprint.pprint(normalization())


# ===========================================================================================================
# This function prints the data in human friendly way to standard output. As input takes distances in format:
# distance = [(dist1, (a1, ..., xn), class), (dist2, (b1, ..., yn), class), ...]
# distance1 = [(dist3, (x1, ..., xn), class), (dist4, (y1, ..., yn), class), ...]
# distance2 = [(dist5, (c1, ..., xn), class), (dist6, (d1, ..., yn), class), ...]
# and dimension of points.
# ===========================================================================================================
def printNeighbours(distance, distance1, distance2, dim):
    print("Euclidean: ", end='')
    print(dim * "\t", end='')
    print("Mink/Cheb: ", end='')
    print(dim * "\t", end='')
    print("Chi2: ")

    for i in range(len(distance)):
        print(distance[i], end='')
        print("\t", end='')
        print(distance1[i], end='')
        print("\t", end='')
        print(distance2[i])


# This function implements calculations necessary for determination of distance using Chi Square formula
def distanceChiSquare(dataSet, pkt, dim):
    accumulator = 0
    for d in range(dim):
        accumulator += ((dataSet[d] - pkt[d]) ** 2) / (dataSet[d] + pkt[d])

    return 0.5 * accumulator


# This function implements calculations necessary for determination of distance using Minkovsky formula
def distanceMinkovsky(dataSet, p, dim, power=10):
    accumulator = 0
    for d in range(dim):
        accumulator += (dataSet[d] - p[d]) ** power

    return accumulator ** (1 / float(power))


# This function implents calcualtions necessary for determination of euclidean distance. Followed formula
def distanceEuclidean(data_set, pkt, dim):
    accumulator = 0
    for d in range(dim):
        accumulator += (data_set[d] - pkt[d]) ** 2

    return math.sqrt(accumulator)


# =======================================================================================================
# This function takes as input a sorted by distance vector which contains k nearest neighbours in format:
# distance = [(dist1, (x1, ..., xn), class), (dist2, (y1, ..., yn), class), ...]
# and checks occurence of classes 0 and 1.
# =======================================================================================================
def result(distance):
    freq1 = 0
    freq2 = 0
    for d in distance:
        if d[2] == 0:
            freq1 += 1
        else:
            freq2 += 1

    return 0 if freq1 > freq2 else 1


# ============================================================================================================
# This function prepares data for final classification and returns the result. First we determine with what
# dimension we are dealing with. Then we create empty lists for three different distance: euclidean, minkovsky
# and chi square. Later two nested loops iterate over classified points in points variable so that functions
# distanceEuclidean, distanceMinkovsky and distanceChiSquare can measure the distance to each point.
# Calculated distances are rounded to two decimal points and appended to previoulsy mentioned lists.
# Then we print all distances to all points. Further we sort all lists by distance in ascending order
# and take first k positions. At last all k nearest points classification is checked by result function
# and the result is returned.
# ============================================================================================================
def classifyAPoint(p, points = pointsXD, k=4, printNeighbouring = "No", printAll = "No", printThreeResults = "No"):

    dim = len(p)
    distanceEucl = []
    distanceMink = []
    distanceChi2 = []

    for group in points:
        for feature in points[group]:

            distanceE = distanceEuclidean(feature, p, dim)
            distanceE = round(distanceE,  2)
            distanceEucl.append((distanceE, feature, group))

            distanceM = distanceMinkovsky(feature, p, dim, 10)
            distanceM = round(distanceM, 2)
            distanceMink.append((distanceM, feature, group))

            distanceC = distanceChiSquare(feature, p, dim)
            distanceC = round(distanceC, 2)
            distanceChi2.append((distanceC, feature, group))


    if printAll == "Yes":

        print("Distance to all poits:")
        printNeighbours(distanceEucl, distanceMink, distanceChi2, dim)
        print(20 * "-")


    if printNeighbouring == "Yes":

        distanceEucl = sorted(distanceEucl)[:k]
        distanceMink = sorted(distanceMink)[:k]
        distanceChi2 = sorted(distanceChi2)[:k]

        print("Distance to k = %s nearest points: " % k)
        printNeighbours(distanceEucl, distanceMink, distanceChi2, dim)

    resultE = result(distanceEucl)
    resultM = result(distanceMink)
    resultC = result(distanceChi2)

    if printThreeResults == "Yes":

        print("Euclid classified point %s as %d, Minkovsky classified as %d and Chi2 inventor classified as %d " % (p, resultE, resultM, resultC))

    return resultE


def executionAnalysis(kEnd=50, mod=1000, dim=10):
    # Time vs k, fixed: dim = 10, points = every 1000
    kValues = [*range(1, kEnd + 1, 1)]
    execTimesK = []
    pointsXD = getPoints(dim, mod)
    p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

    for k in kValues:
        startTime = time.clock()
        classifyAPoint(p, pointsXD, k)
        timeX = time.clock() - startTime
        execTimesK.append(timeX)

    plt.subplot(131)
    plt.plot(kValues, execTimesK, 'o', color='red')
    plt.show()
    # Dim vs Time, fixed: k = 7, points = every 1000
    dValues = [*range(1, dim + 1, 1)]
    execTimesDim = []
    dimensions = ["%dD" % (i + 1) for i in range(dim)]
    k = 7

    for d in dValues:
        pointsXD = getPoints(d, mod)
        p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

        start_time1 = time.clock()
        classifyAPoint(p, pointsXD, k)
        time1 = time.clock() - start_time1
        execTimesDim.append(time1)

    plt.subplot(132)
    plt.plot(dimensions, execTimesDim, 'o', color='green')
    plt.show()
    # Points vs Time, fixed: k = 7, dim = 10
    mValues = [*range(0, 10000 + 11, 500)]
    mValues.pop(0)
    execTimesP = []
    k = 7

    for m in mValues:
        pointsXD = getPoints(dim, m)
        p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

        start_time1 = time.clock()
        classifyAPoint(p, pointsXD, k)
        time1 = time.clock() - start_time1
        execTimesP.append(time1)

    plt.subplot(133)
    plt.plot(mValues, execTimesP, 'o', color='blue')
    plt.show()

executionAnalysis()

# =====================================================================================
# Firstly for plots 2D there are definition of the lists which in the next step will be
# filled in and they will contain x or y features of points belonging to class  0 or 1
# from training set. It is used for draw all points on the plot.

# plt.subplot() --> divides the plot into subplots
# plt.plot() --> draws plot
# plt.xlabel() --> adds label
# plt.grid() --> turns on a grid on plots
# plt.tight_layout() --> create bigger gap between subplots.
# =====================================================================================
tempDict = getPoints(2, 10000)
tempDict = normalization(tempDict)

p = [0.9, 0.9]

rc = classifyAPoint(p, tempDict, 3)

x_0_2D = [tempDict[0][x][0] for x in range(len(tempDict[0]))]
x_1_2D = [tempDict[1][x][0] for x in range(len(tempDict[1]))]
y_0_2D = [tempDict[0][y][1] for y in range(len(tempDict[0]))]
y_1_2D = [tempDict[1][y][1] for y in range(len(tempDict[1]))]

if rc == 0:
    col = 'red'
elif rc == 1:
    col = 'green'

plt.subplot(121)
plt.plot(x_0_2D, y_0_2D, 'o', color='red')
plt.plot(x_1_2D, y_1_2D, 'o', color='green')
plt.plot(p[0], p[1], 'o', color='blue')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True)
plt.show()
plt.subplot(122)
plt.plot(x_0_2D, y_0_2D, 'o', color='red')
plt.plot(x_1_2D, y_1_2D, 'o', color='green')
plt.plot(p[0], p[1], 'o', color=col)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True)

plt.tight_layout(pad=3.0)

plt.show()

# Comparison with scikitlearn

X = tempDict[0] + tempDict[1]
X = np.array([np.array(i) for i in X])
y = [0] * len(tempDict[0]) + [1] * len(tempDict[1])

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

clf = neighbors.KNeighborsClassifier(3, weights='uniform')
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = %i, weights = '%s')"
          % (3, 'uniform'))

# ==================================================================================
# For plots 3D the steps are analogous. There are definition of the lists which in
# the next step will be filled in and they will contain x, y or z features of points
# belonging to class 0 or 1 from training set. Then the plots are drawn.
# ==================================================================================
tempDict = getPoints(3, 5000)
tempDict = normalization(tempDict)

p = [0.5, 0.5, 0.5]

rc = classifyAPoint(p, tempDict, 3)

x_0_3D = [tempDict[0][x][0] for x in range(len(tempDict[0]))]
x_1_3D = [tempDict[1][x][0] for x in range(len(tempDict[1]))]
y_0_3D = [tempDict[0][y][1] for y in range(len(tempDict[0]))]
y_1_3D = [tempDict[1][y][1] for y in range(len(tempDict[1]))]
z_0_3D = [tempDict[0][z][2] for z in range(len(tempDict[0]))]
z_1_3D = [tempDict[1][z][2] for z in range(len(tempDict[1]))]

if rc == 0:
    col = 'red'
elif rc == 1:
    col = 'green'

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter3D(x_0_3D, y_0_3D, z_0_3D, c='red')
ax1.scatter3D(x_1_3D, y_1_3D, z_1_3D, c='green')
ax1.scatter3D(p[0], p[1], p[2], c='blue')

plt.show()

fig = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.scatter3D(x_0_3D, y_0_3D, z_0_3D, c='red')
ax2.scatter3D(x_1_3D, y_1_3D, z_1_3D, c='green')
ax2.scatter3D(p[0], p[1], p[2], c=col)
plt.show()