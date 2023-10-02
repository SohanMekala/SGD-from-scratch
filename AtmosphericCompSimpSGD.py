import pandas as pd
import matplotlib.pyplot as plt
import numpy as np #ONLY FOR TESTING PURPOSES
import random

data = pd.read_csv('ABoVE_april-nov_2017_flask_data.csv')#read data and store as pdframe
selCols = [i for i in range(6)]+[8,9,11]+[i for i in range(16, 221, 4)]#take only temporal data, geographical data, altitude, and concentration of atmospherical elements(in ppt, ppb, and ppm)
data = data.iloc[:,selCols]#only take selected cols
data = data.sort_values(by='sample_elevation', ascending=True)#sort by elevation vals

def cleanData(element): #get rid of missing elevations and duplicate elevations
    x = []

    for i in range (data.shape[0]):
        if(data.iloc[i,data.columns.get_loc(element)]!=-999.99):
            x.append(data.iloc[i,data.columns.get_loc('sample_elevation')])
            
    y = [val for val in data[element] if val != -999.99]
    
    x2 = [x[0]] #remove duplicate elevations
    y2 = [y[0]]
    
    rawElevationsCount = len(x)
    currentElevation = x2[0]
    
    for i in range(rawElevationsCount):
        if(x[i]!=currentElevation):
            x2.append(x[i])
            y2.append(y[i])
            currentElevation = x[i]
        else:
            i+=1
    
    return(x2,y2)


def plotData(x,y,x1, y1,elementName, concentrationMeasurment): #plot it
    plt.scatter(x, y,)
    plt.plot(x1, y1, linewidth=5,color = "black")
    plt.xlabel('elevation(meters)')
    plt.ylabel(elementName+' concentration('+concentrationMeasurment+')')
    plt.title(elementName+' concentration by elevation')
    plt.grid(True)
    plt.show()
    

e = 'CH4_analysis_value'
en = 'methane'
mes = "ppb"

methaneX, methaneY = cleanData(e) 

e1 = 'CO2_analysis_value'
en1 = 'carbon dioxide'
mes1 = "ppm"

carbondioxideX, carbondioxideY = cleanData(e1)

#in this case I am taking methane and carbondioxide for analysis

def meanSquaredError(m,b,xVals,yVals): #simple MSE function
    totalError = 0
    for i in range(len(xVals)):
        x = xVals[i]
        y = yVals[i]
        totalError += (y - (m * x + b))**2
            
    return totalError/float(len(xVals))

def stochastic_gradient_descent(xVals, yVals, learning_rate, num_epochs):
    # Initialize weights (slope) and bias (intercept) to random values
    m = random.random()
    b = random.random()
    
    n = len(xVals)
    
    for epoch in range(num_epochs):
        for i in range(n):
            #choosing not to go random because that takes too long
            x = xVals[n-1-i] #go backwards because data is skewed right
            y = yVals[n-1-i]
            
            # Calculate the predicted value
            y_pred = m * x + b
            
            # Calculate the gradients of the loss function with respect to m and b
            m_gradient = -2 * x * (y - y_pred) #got rid of N because function was too volatile
            b_gradient = -2 * (y - y_pred)
            
            # Update the weights and bias using the gradients and learning rate
            m -= (learning_rate * m_gradient)
            b -= (learning_rate * b_gradient)
    
    return m, b

learning_rate = 0.000001 #got to these numbers after a lot of testing
num_epochs = 500 #because this is a simple SGD, i am still yet to add ways to validate, prevent overfitting and underfitting, and deal with large outliers

#x = methaneX
#y = methaneY
x = carbondioxideX
y = carbondioxideY

m,b= stochastic_gradient_descent(x, y, learning_rate, num_epochs)

m1,b1 = np.polyfit(x,y,1) #lets see polyfits example
print("MSE difference between mine and numpy's: "+str(meanSquaredError(m,b,x,y)-meanSquaredError(m1,b1,x,y)))

LOBF_x = [i for i in range(2000)] #plotting purpsoes
LOBF_y = [i*m+b for i in range(2000)] #plotting purposes

plotData(x,y,LOBF_x,LOBF_y,en1,mes1)
