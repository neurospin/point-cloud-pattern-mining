#! /usr/bin/env python

import numpy as np
from scipy import linalg


class Icp:
    """ Iterative Closest Point """

    def __init__(self, arg1):
        self.arg1 = arg1

    def __init__(self, inData, inModel, iteLim, errLim):
        self.data = inData
        self.model = inModel
        self.iteLim = iteLim
        self.errLim = errLim
        self.numPtr = self.data.shape[1]
        self.numDim = self.data.shape[0]

        self.minDist = np.zeros((self.numPtr, 2), 'f')
        self.curClose = inData.copy()
        self.curData = inData.copy()
        self.cumRot = np.eye(self.numDim, self.numDim)
        self.cumTrans = np.zeros((self.numDim, 1), 'f')
        self.curR = np.eye(self.numDim, self.numDim)
        self.curT = np.zeros((self.numDim, 1), 'f')
        self.curDist = 10000.0
        self.transModel = self.model.T

        # kdtree
        #self.kmodel = sp.spatial.cKDTree(self.model.T)

    def calcClose(self):
        """ calculate the closest model point, given data """

        temp = self.model.T[..., None]
        dist = ((temp - self.curData)**2).sum(axis=1)
        #print("in calcClose..........")
        # print(dist)
        indices = dist.argmin(axis=0)
        self.minDist[:, 1] = dist.min(axis=0)
        self.minDist[:, 0] = indices
        self.curClose = self.model[:, indices]
        self.curDist = self.minDist[:, 1].sum()/self.numPtr

    def calcTransform(self):
        """ calculate the rotation and translation matrix """

        meanModel = self.model.mean(axis=1)[:, None]
        meanData = self.curData.mean(axis=1)[:, None]
        A = self.curData - meanData
        B = self.curClose - meanModel
        (U, S, V) = linalg.svd(np.dot(B, A.T))
        U[:, -1] *= linalg.det(np.dot(U, V))
        self.curR = np.dot(U, V)
        self.curT = meanModel - np.dot(self.curR, meanData)

    def calcIcp(self):
        """ 
        calculate the distance by ICP(Iterative Closest Point)
        input: model, data, ite
        output: the distance between data and model  """

        ite = 0
        err = 10000
        preDist = err
        while ((ite < self.iteLim) and (err > self.errLim)):

            self.calcClose()
            self.calcTransform()
            self.curData = np.dot(self.curR, self.curData)
            self.curData = self.curData + self.curT
            self.cumRot = np.dot(self.curR, self.cumRot)
            self.cumTrans = np.dot(self.curR, self.cumTrans) + self.curT
            err = abs(preDist - self.curDist)
            ite = ite + 1
            preDist = self.curDist
        return self.curDist, self.cumRot, self.cumTrans

    def calcSimple(self):
        """ 
        calculate simple distance without ICP
        input: model, data
        output: the distance between data and model  """

        self.calcClose()
        return self.curDist, self.cumRot, self.cumTrans
