#���� �ٲ��
from numpy import *
import operator
from collections import Counter

def createDataSet():
    group = array([ [1.0, 2.0], [1.0,4.0], [4.0, 1.0], [4.0, 2.0] ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def calcDistance(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]  # dataSetSize = 4
    #tile �� �ݺ���� ���� �� ���
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # **2 means square
    sqDiffMat = diffMat ** 2

    # sqDistances = x^2 + y^2
    sqDistances = sqDiffMat.sum(axis=1)
    # distance is equal to the square root of the sum of the squares of the coordinates
    # distance = [2, 2, 8, 5]
    distances = sqDistances ** 0.5
    
    
    # here returns [0 1 3 2]
    # argsort �� ��Ʈ�� ������ index�� ���·� ��ȯ�Ѵ�.
    sortedDistIndices = distances.argsort()
    
    return sortedDistIndices

def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
    #classCount = {}
    a = []
    # argsort�� sort�� ������ �󺧷� �����ֱ� ���� ���
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        a.append(voteIlabel)
        
    #counter �� k������ ������ ���� ã������
    count = Counter(a)
    
    #counter �� ���´� Counter({'romantic:2,'action':1}) �����ε� label ���� ������ ���ؼ� [0][0] ���
    result = count.most_common(1)[0][0]
    

    return result


group, labels = createDataSet() #group = ������ �ִ� �����ͼ� , label = �����ͼ��� �󺧵�
sortedDistIndices = calcDistance([4.0, 3.0], group, labels, 3)
result = findMajorityClass([4.0, 3.0], group, labels,3,sortedDistIndices) # ���ο� ������, ���� ������, �� , k=3
print(result)
