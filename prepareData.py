# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:25:23 2023

@author: Agnieszka Florkowska
"""
import os
import pandas as pd
import numpy as np
import random
import operator
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random
import cv2
from skimage.feature import local_binary_pattern

TEST_SET_SIZE = 0.3 # 1 - TEST_SET_SIZE (70%) - training set, TEST_SET_SIZE (30%) testing set
VAL_SET_SIZE = 0.5  # 50% of testing set, it means     
                    # TEST_SET_SIZE / 2 (15%) - testing set, TEST_SET_SIZE / 2 (15%) validation set

def shuffleData(set1, set2):
    temp = list(zip(set1, set2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return res1,res2

def createPath(mainPath, patAndSlice):
    if patAndSlice[0] < 100:
        patientNumString = '0' + str(patAndSlice[0])
    else:
        patientNumString = str(patAndSlice[0])
    finalPath = mainPath + '\Patients_CT\\' + patientNumString + '\\brain\\' + str(patAndSlice[1]) + '.jpg'    
    return finalPath

def swap_target(x):
    if x == 0:
        return 1
    else:
        return 0

class DatasetSplittingType(Enum):
    kFOLD= 0
    TRAIN_TEST = 1
    TRAIN_VAL_TEST = 2

def plotHistogram(data, title, xlabel, ylabel, color = '#0504aa'):
    figure(figsize=(10, 8), dpi=300)
    n, bins, patches = plt.hist(x=data, bins='auto', color=color, alpha=0.7, rwidth=0.85)
    plt.grid(axis='both', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def piePlot(data, legend, title):
    figure(figsize=(10, 8), dpi=300)
    plt.pie(data, labels = legend, autopct='%.2f') 
    plt.grid(axis='both', alpha=0.75)
    plt.title(title)
    plt.show() 
    
class HemorrageDataset:
    def __init__(self, diagnoseCsvPath, demographyCsvPath, folderPath):
        self.__diagnoseCSV = pd.read_csv(diagnoseCsvPath)
        self.__demographyCsv = pd.read_csv(demographyCsvPath)
        self.__pathToWholeFolder = folderPath
        self.__trainDataForLoading = []
        self.__trainLabelsForLoading = []
        self.__testDataForLoading = []
        self.__testLabelsForLoading = []
        self.__valDataForLoading = []
        self.__valLabelsForLoading = []
        self.__kFoldDataForLoading = []
        self.__kFoldLabelsForLoading = []
      
    def __kFoldSplitting(self, k, sickCases, healthyCases):
        healthyLabels = [1 for i in range(len(healthyCases))]
        sickLabels = [0 for i in range(len(sickCases))]
        allCases = healthyCases + sickCases
        allLabels = healthyLabels + sickLabels
        allCases, allLabels = shuffle(allCases, allLabels)
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
        for train_index, test_index in skf.split(allCases, allLabels):
            trainNumbersFold, testNumbersFold = operator.itemgetter(*train_index)(allCases), operator.itemgetter(*test_index)(allCases)
            trainData, trainLabels = self.__prepareDataSavingPatienNumberAndSlice(trainNumbersFold)
            testData, testLabels = self.__prepareDataSavingPatienNumberAndSlice(testNumbersFold)
            self.__kFoldDataForLoading.append([trainData, testData])
            self.__kFoldLabelsForLoading.append([trainLabels, testLabels])

    
    def __subsetSplitting(self, sickCases, healthyCases):
        trainHealthyPatientsNumbers, testHealthyPatientsNumbers = train_test_split(healthyCases, test_size=TEST_SET_SIZE,random_state=25, shuffle=True)
        testHealthyPatientsNumbers, valHealthyPatientsNumbers = train_test_split(testHealthyPatientsNumbers, test_size=VAL_SET_SIZE,random_state=25, shuffle=True)
        trainSickPatientsNumbers, testSickPatientsNumbers = train_test_split(sickCases, test_size=TEST_SET_SIZE,random_state=25, shuffle=True)
        testSickPatientsNumbers, valSickPatientsNumbers = train_test_split(testSickPatientsNumbers, test_size=VAL_SET_SIZE,random_state=25, shuffle=True)
        
        trainCases = trainHealthyPatientsNumbers + trainSickPatientsNumbers 
        trainCases = random.sample(trainCases, len(trainCases))
        testCases = testHealthyPatientsNumbers + testSickPatientsNumbers
        testCases = random.sample(testCases, len(testCases))
        valCases = valHealthyPatientsNumbers + valSickPatientsNumbers
        valCases = random.sample(valCases, len(valCases))
        return trainCases, testCases, valCases
    
    def __distinquishHealthyAndSickCases(self):
        sickCases = []
        healthyCases = []
        for patientNum in np.unique(self.__diagnoseCSV['PatientNumber']):
            isSick = self.__diagnoseCSV[(self.__diagnoseCSV['PatientNumber'] == patientNum)].Has_Hemorrhage.sum()
            if isSick > 0:
                sickCases.append(patientNum)
            else:
                healthyCases.append(patientNum)
                
        return healthyCases, sickCases

    def __prepareDataSavingPatienNumberAndSlice(self, chosenSet):
        data = []
        labels = []
        for patientNum in chosenSet:
            for sliceNum in np.unique(self.__diagnoseCSV.loc[(self.__diagnoseCSV['PatientNumber'] == patientNum)]['SliceNumber']):
                diagnose = self.__diagnoseCSV.loc[(self.__diagnoseCSV['PatientNumber'] == patientNum) 
                                        & (self.__diagnoseCSV['SliceNumber'] == sliceNum)]['Has_Hemorrhage'].values[0]
                data.append((patientNum, sliceNum))
                labels.append(diagnose)
                
        return shuffleData(data, labels)
    
    def splitDatasetBasedOnPatientsCases(self, splittingType, kFold = 0):
         healthyPatientsNumbers, sickPatientsNumbers = self.__distinquishHealthyAndSickCases()
         
         if(splittingType == DatasetSplittingType.kFOLD):
             self.__kFoldSplitting(kFold, sickPatientsNumbers, healthyPatientsNumbers)
         elif ((splittingType == DatasetSplittingType.TRAIN_VAL_TEST) or (splittingType == DatasetSplittingType.TRAIN_TEST)):
             trainSubset, testSubset, valSubset = self.__subsetSplitting(sickPatientsNumbers, healthyPatientsNumbers) 
             self.__trainDataForLoading, self.__trainLabelsForLoading = self.__prepareDataSavingPatienNumberAndSlice(trainSubset)
             self.__testDataForLoading, self.__testLabelsForLoading = self.__prepareDataSavingPatienNumberAndSlice(testSubset)
             self.__valDataForLoading, self.__valLabelsForLoading = self.__prepareDataSavingPatienNumberAndSlice(valSubset)
             
             if(splittingType == DatasetSplittingType.TRAIN_TEST):
                 self.__testDataForLoading += self.__valDataForLoading
                 self.__testLabelsForLoading += self.__valLabelsForLoading
                 self.__valDataForLoading = []
                 self.__valLabelsForLoading= []
                         
    def removeRecordFromDataset(self, patientNum, sliceNumber):
        # Remove corrupted images or with missing brain/bone windowed images
        index_to_drop = self.__diagnoseCSV[(self.__diagnoseCSV['PatientNumber'] == patientNum) & (self.__diagnoseCSV['SliceNumber'] == sliceNumber)].index
        index_to_drop = index_to_drop[0]
        self.__diagnoseCSV = self.__diagnoseCSV.drop(index_to_drop, axis=0)  
        
        
    def invertBinaryValues(self, colToChangeAndRemove, newCol):
        self.__diagnoseCSV[newCol] = self.__diagnoseCSV[colToChangeAndRemove].apply(swap_target)
        self.__diagnoseCSV = self.__diagnoseCSV.drop(colToChangeAndRemove, axis=1)    
    
    def get_trainDataWithLabels(self):
        return self.__trainDataForLoading, self.__trainLabelsForLoading

    def get_testDataWithLabels(self):
        return self.__testDataForLoading, self.__testLabelsForLoading
    
    def get_valDataWithLabels(self):
        return self.__valDataForLoading, self.__valLabelsForLoading
    
    def get_kFoldDataWithLabels(self):
        return self.__kFoldDataForLoading, self.__kFoldLabelsForLoading

    def countHealthyAndSickSliceDistribution(self):
        diagnosed = self.__diagnoseCSV['Has_Hemorrhage'].value_counts()
        return diagnosed 
  
    def countGenderDistribution(self):
        gender = self.__demographyCsv['Gender'].value_counts()
        return gender
    
    def countAgeDistribution(self):
        age = self.__demographyCsv['Age\n(years)']
        return age


         
# Create class for ml method - load images, preprocess, choose method and fit model 
# another class for statistic and visualization -> acces private property: self._Parent__private(), self.__demographyCSV = pd.read_csv(demographyCsvPath)
# generator for all
# for imbalanced dataset: https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/

#########################################         
basePath  = r'D:\Brain_JPG_Hemmorage\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0'
diagnoseCsvPath = basePath + '\hemorrhage_diagnosis.csv'
demographyCsvPath = basePath + '\patient_demographics.csv'
images = basePath + '\Patients_CT'
dataset = HemorrageDataset(diagnoseCsvPath, demographyCsvPath, basePath) 

# Prepare csv file  
dataset.removeRecordFromDataset(84, 36)
dataset.invertBinaryValues('No_Hemorrhage', 'Has_Hemorrhage')

'''
# Analyse Dataset
diagnosedSlices = dataset.countHealthyAndSickSliceDistribution()
piePlot(diagnosedSlices, ['Normalne', 'Z krwotokiem'], "Procentowy rozkład przekrojów w zbiorze")
genderCounter = dataset.countGenderDistribution()
piePlot(genderCounter, ['Mężczyźni', 'Kobiety'], "Procentowy rozkład płci w zbiorze")
ageCounter = dataset.countAgeDistribution()
plotHistogram(ageCounter, "Rozkład wieku", "Wiek", "Liczba przypadków")
'''

# Split dataset using chosen method
####  2/3 sets splitting
dataset.splitDatasetBasedOnPatientsCases(DatasetSplittingType.TRAIN_VAL_TEST)
trainData, trainLabels = dataset.get_trainDataWithLabels()
testData, testLabels = dataset.get_testDataWithLabels()
valData, valLabels = dataset.get_valDataWithLabels()

# ####  stratified k - fold
# dataset.splitDatasetBasedOnPatientsCases(DatasetSplittingType.kFOLD, 10)
# kfoldData, kfoldLabel = dataset.get_kFoldDataWithLabels()

for i in range(0,10):
    tempPath = createPath(basePath, trainData[i])
    image = cv2.imread(tempPath)
    plt.imshow(image)
    plt.show()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numPoints = 8 
    radius = 1
    lbp = local_binary_pattern(gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),  bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    plt.plot(hist)
