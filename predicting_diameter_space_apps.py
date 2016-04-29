# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 05:24:36 2016
@author: Tautvydas Mickus


"""
import scipy
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from lxml import objectify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import math
import sys, requests
import json
import ast
import yaml



#request asteroid data that contains these parameters:
params={}
#params["number"]="267221"
params["json"]="1"
params["diameter_neowise_min"]="1"
params["diameter_neowise_max"]="10"
params["albedo_neowise_min"]="0"
##params["g_neowise_min"]="0"
#params["h_neowise_min"]="0"
params["spin_period_min"]="0"
params["eccentricity_min"]="0"
params["period_min"]="0"
#params["phase_slope_min"]="0"
params["absolute_magnitude_min"]="0"
#orbit typeearth_moid
#params["earth_moid_min"]="0"


url = 'http://mpc.cfa.harvard.edu/ws/search'
r = requests.post(url, params, auth = ('mpc_ws', 'mpc!!ws'))

# convert request to proper 'matrix' format
json_data=json.loads(r.text)
json_dumped_data=json.dumps(json_data)
asteroid_dictionary=yaml.safe_load(json_dumped_data)


# initialise array ('matrix') with first values
diameter_neowise = asteroid_dictionary[0]['diameter_neowise']
albedo_neowise=asteroid_dictionary[0]['albedo_neowise']
spin_period=asteroid_dictionary[0]['spin_period']
eccentricity=float(asteroid_dictionary[0]['eccentricity'])
period=float(asteroid_dictionary[0]['period'])
orbit_type=float(asteroid_dictionary[0]['orbit_type'])
absolute_magnitude=float(asteroid_dictionary[0]['absolute_magnitude'])
semimajor_axis=float(asteroid_dictionary[0]['semimajor_axis'])
aphelion_distance=float(asteroid_dictionary[0]['aphelion_distance'])
perihelion_distance =float(asteroid_dictionary[0]['perihelion_distance'])
argument_of_perihelion =float(asteroid_dictionary[0]['argument_of_perihelion'])
ascending_node =float(asteroid_dictionary[0]['ascending_node'])
inclination =float(asteroid_dictionary[0]['inclination'])

asteroid_data = np.array([[albedo_neowise, diameter_neowise,spin_period,eccentricity,period,orbit_type,absolute_magnitude,semimajor_axis,aphelion_distance,perihelion_distance,argument_of_perihelion,ascending_node,inclination]])

# loop through the requested data and save it appropriately in a np.array so that could be used easier later
for i in range(1,len(asteroid_dictionary)):
    #get all data into the correct matrix format    
    diameter_neowise = asteroid_dictionary[i]['diameter_neowise']
    albedo_neowise=asteroid_dictionary[i]['albedo_neowise']
   
    spin_period=asteroid_dictionary[i]['spin_period']
    eccentricity=float(asteroid_dictionary[i]['eccentricity'])
    period=float(asteroid_dictionary[i]['period'])
    orbit_type=float(asteroid_dictionary[i]['orbit_type'])
    absolute_magnitude=float(asteroid_dictionary[i]['absolute_magnitude'])
    semimajor_axis=float(asteroid_dictionary[i]['semimajor_axis'])
    aphelion_distance=float(asteroid_dictionary[i]['aphelion_distance'])
    perihelion_distance =float(asteroid_dictionary[i]['perihelion_distance'])
    argument_of_perihelion =float(asteroid_dictionary[i]['argument_of_perihelion'])
    ascending_node =float(asteroid_dictionary[i]['ascending_node'])
    inclination =float(asteroid_dictionary[i]['inclination'])
    this_asteroid_data = np.array([[albedo_neowise, diameter_neowise,spin_period,eccentricity,period,orbit_type,absolute_magnitude,semimajor_axis,aphelion_distance,perihelion_distance,argument_of_perihelion,ascending_node,inclination]])     
    asteroid_data=np.concatenate((asteroid_data, this_asteroid_data), axis=0)

print("Total number of samples: ", str(len(asteroid_dictionary)))
    
# 2/3 data is used as training set, 1/3 as testing set
train_points=int((len(asteroid_dictionary)-10)/3*2) 
test_points=int((len(asteroid_dictionary)-10)/3*1)

#create dataframe and give the column names
asteroid_dataframe=pd.DataFrame(asteroid_data)
column_names=['albedo_neowise','diameter_neowise','spin_period','eccentricity','period','orbit_type','absolute_magnitude','semimajor_axis','aphelion_distance','perihelion_distance','argument_of_perihelion','ascending_node','inclination']
asteroid_dataframe.columns=column_names


#drop the target set and also the data that practically wouldnt be known (albedo)
x=asteroid_dataframe.drop('diameter_neowise',axis = 1)
x=x.drop('albedo_neowise',axis = 1)


#create teaching and training sets
y=asteroid_dataframe[['diameter_neowise']]
train_targets=y.ix[0:train_points-1,0]
test_target=y.ix[train_points:train_points+test_points,0]

train_set=x.ix[0:train_points-1,:]
test_set=x.ix[train_points:train_points+test_points,:]


#Linear regression
#lm = LinearRegression()
#lm.fit(train_set,train_targets)


#using Bayesian Linear Regression train algorithm based on training set
clf = linear_model.BayesianRidge()
clf.fit(train_set,train_targets)


#display coefficient data for info
coefficientData=pd.DataFrame(zip(x.columns,lm.coef_),columns=['Features', 'Estimated Coefficients'])
print coefficientData
predicted_targets=clf.predict(test_set)

plt.close("all")
plt.figure(1)
plt.scatter(test_target,predicted_targets)
plt.xlabel("real diameter (km) ")
plt.ylabel("predicted diameter (km)")
plt.title("Figure 1b: Scatter plot: \n Real diameter vs predicted diameter")
lim=10
plt.xlim([1,lim])
plt.ylim([1,lim])
plt.plot([1, 10], [1, 10], color='r', linestyle='-', linewidth=2)
mseFull = np.mean((test_target-predicted_targets)**2)
print "MSE of predicted values using the machine learning (Bayesian Linear Regression):" , mseFull
#
#plt.figure()
#plt.scatter(lm.predict(train_set), lm.predict(train_set)-train_targets, c='b', s=40, alpha=0.5)
#plt.scatter(lm.predict(test_set), predicted_targets - test_target, c='g', s=40)
#plt.hlines(y=0, xmin=0, xmax=10)



#get absolute magnitude data
allh=asteroid_dataframe[['absolute_magnitude']]
allh=allh.ix[:,0]


targets_calculated=[]


for i in range(0,test_points+1):
    #assume albedo is on maximum range, calculate the diameter
    p=0.05
    H=allh[1000+i]
    diameter=(1329/math.sqrt(p))*math.pow(10,-0.2*H)
    targets_calculated.append(diameter)
#plot the figure for albedo with maximum range
plt.figure(2)
#plt.scatter(test_target,targets_calculated,c='r')
mseFull = np.mean((test_target-targets_calculated)**2)
print "MSE of predicted values using assumed albedo of 0.05:" ,mseFull
targets_calculated=[]


for i in range(0,test_points+1):
    #assume albedo is minimum range
    p=0.15
    H=allh[train_points+i]
    diameter=(1329/math.sqrt(p))*math.pow(10,-0.2*H)
    targets_calculated.append(diameter)
#plot the figure for albedo with maximum range

plt.scatter(test_target,targets_calculated,c='g')
plt.plot([1, 10], [1, 10], color='r', linestyle='-', linewidth=2)

mseFull = np.mean((test_target-targets_calculated)**2)
print "MSE of predicted values using assumed albedo of 0.15:" ,mseFull
targets_calculated=[]
for i in range(0,test_points+1):
    #assume albedo is on maximum range, calculate the diameter
    p=0.25 
    H=allh[1000+i]
    diameter=(1329/math.sqrt(p))*math.pow(10,-0.2*H)
    targets_calculated.append(diameter)
    
#plt.scatter(test_target,targets_calculated)
plt.xlim([1,lim])
plt.ylim([1,lim])
plt.legend(['albedo value assumed as 0.15'])

plt.xlabel("real diameter (km)")
plt.ylabel("calculated diameter with assumed albedo value (km)")
plt.title("Figure 1a: Scatter Plot of: \n Real diamater vs calculated diameter using assumed albedo value" )

plt.xlim([1,lim])
plt.ylim([1,lim])


mseFull = np.mean((test_target-targets_calculated)**2)
print "MSE of predicted values using assumed albedo of 0.25:" ,mseFull
plt.plot([1, 10], [1, 10], color='r', linestyle='-', linewidth=2)