#과연 바뀔까
from numpy import *
import operator
from collections import Counter

def createDataSet():
    group = array([ [1.0, 2.0], [1.0,4.0], [4.0, 1.0], [4.0, 2.0] ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def calcDistance(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]  # dataSetSize = 4
    #tile 은 반복행렬 만들 때 사용
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # **2 means square
    sqDiffMat = diffMat ** 2

    # sqDistances = x^2 + y^2
    sqDistances = sqDiffMat.sum(axis=1)
    # distance is equal to the square root of the sum of the squares of the coordinates
    # distance = [2, 2, 8, 5]
    distances = sqDistances ** 0.5
    
    
    # here returns [0 1 3 2]
    # argsort 는 솔트된 값들을 index의 형태로 반환한다.
    sortedDistIndices = distances.argsort()
    
    return sortedDistIndices

def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
    #classCount = {}
    a = []
    # argsort로 sort된 값들을 라벨로 맞춰주기 위해 사용
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        a.append(voteIlabel)
        
    #counter 로 k값에서 인접한 라벨을 찾기위해
    count = Counter(a)
    
    #counter 의 형태는 Counter({'romantic:2,'action':1}) 형태인데 label 값만 따오기 위해서 [0][0] 사용
    result = count.most_common(1)[0][0]
    

    return result


group, labels = createDataSet() #group = 이전에 있는 데이터셋 , label = 데이터셋의 라벨들
sortedDistIndices = calcDistance([4.0, 3.0], group, labels, 3)
result = findMajorityClass([4.0, 3.0], group, labels,3,sortedDistIndices) # 새로운 데이터, 기존 데이터, 라벨 , k=3
print(result)
