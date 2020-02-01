#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:37:21 2018

@author: j2
"""

# total acc
def Total_Accuracy(c_matrix):
    diag = np.diag(c_matrix)
    # sumamos 0.0 para que no de la division de la total accuracy 0
    total_diag = np.sum(diag[0:c_matrix.shape[0]-2])+0.0
    total_acc = (total_diag/diag[c_matrix.shape[0]-1])*100
    print '\nTotal accuracy:',total_acc
    return total_acc
 
# user acc
def User_Accuracy(c_matrix,class_names):
    user_acc = np.zeros((c_matrix.shape[0]-1))
    for i in range(0,c_matrix.shape[0]-1):
        total_fila = c_matrix["All"][i+1]
        user_acc[i] = ((c_matrix[i+1][i+1]+0.0)/total_fila)*100
    return user_acc
 
# prod acc
def Producer_Accuracy(c_matrix,class_names):
    prod_acc = np.zeros((c_matrix.shape[0]-1))
    for i in range(0,c_matrix.shape[0]-1):
        total_col = c_matrix[i+1]["All"]
        prod_acc[i] = ((c_matrix[i+1][i+1]+0.0)/total_col)*100
    return prod_acc

def Formateo(folder_data):
    T11_4_jun_ds = gdal.Open(folder_data+'20150604_T11.tif', gdal.GA_ReadOnly)
    T11_4_jun_ds = T11_4_jun_ds.ReadAsArray()
    T11_4_jun = np.zeros((9500,9500,1))
    for b in range(0,T11_4_jun_ds.shape[0]):
        T11_4_jun[b, :, :] = T11_4_jun_ds[b,:].reshape(-1,1)

    T11_15_jun_ds = gdal.Open(folder_data+'20150615_T11.tif', gdal.GA_ReadOnly)
    T11_15_jun_ds = T11_15_jun_ds.ReadAsArray()
    T11_15_jun = np.zeros((9500,9500,1))
    for b in range(0,T11_15_jun_ds.shape[0]):
        T11_15_jun[b, :, :] = T11_15_jun_ds[b,:].reshape(-1,1) 

    T11_26_jun_ds = gdal.Open(folder_data+'20150626_T11.tif', gdal.GA_ReadOnly)
    T11_26_jun_ds = T11_26_jun_ds.ReadAsArray()
    T11_26_jun = np.zeros((9500,9500,1))
    for b in range(0,T11_26_jun_ds.shape[0]): 
        T11_26_jun[b, :, :] = T11_26_jun_ds[b,:].reshape(-1,1)  
   
    T11_7_jul_ds = gdal.Open(folder_data+'20150707_T11.tif', gdal.GA_ReadOnly)
    T11_7_jul_ds = T11_7_jul_ds.ReadAsArray()
    T11_7_jul = np.zeros((9500,9500,1))
    for b in range(0,T11_7_jul_ds.shape[0]):
        T11_7_jul[b, :, :] = T11_7_jul_ds[b,:].reshape(-1,1) 

    T11_18_jul_ds = gdal.Open(folder_data+'20150718_T11.tif', gdal.GA_ReadOnly)
    T11_18_jul_ds = T11_18_jul_ds.ReadAsArray()
    T11_18_jul = np.zeros((9500,9500,1))
    for b in range(0,T11_18_jul_ds.shape[0]):
        T11_18_jul[b, :, :] = T11_18_jul_ds[b,:].reshape(-1,1)
        
    T11_29_jul_ds = gdal.Open(folder_data+'20150729_T11.tif', gdal.GA_ReadOnly)
    T11_29_jul_ds = T11_29_jul_ds.ReadAsArray()
    T11_29_jul = np.zeros((9500,9500,1))
    for b in range(0,T11_29_jul_ds.shape[0]):
        T11_29_jul[b, :, :] = T11_29_jul_ds[b,:].reshape(-1,1)
        
    T11_9_ago_ds = gdal.Open(folder_data+'20150809_T11.tif', gdal.GA_ReadOnly)
    T11_9_ago_ds = T11_9_ago_ds.ReadAsArray()
    T11_9_ago = np.zeros((9500,9500,1))
    for b in range(0,T11_9_ago_ds.shape[0]):
        T11_9_ago[b, :, :] = T11_9_ago_ds[b,:].reshape(-1,1)

    T11_20_ago_ds = gdal.Open(folder_data+'20150820_T11.tif', gdal.GA_ReadOnly)
    T11_20_ago_ds = T11_20_ago_ds.ReadAsArray()
    T11_20_ago = np.zeros((9500,9500,1))
    for b in range(0,T11_20_ago_ds.shape[0]):
        T11_20_ago[b, :, :] = T11_20_ago_ds[b,:].reshape(-1,1)

    T11_31_ago_ds = gdal.Open(folder_data+'20150831_T11.tif', gdal.GA_ReadOnly)
    T11_31_ago_ds = T11_31_ago_ds.ReadAsArray()
    T11_31_ago = np.zeros((9500,9500,1))
    for b in range(0,T11_31_ago_ds.shape[0]):
        T11_31_ago[b, :, :] = T11_31_ago_ds[b,:].reshape(-1,1)

    
    # lectura y formateo T22
    T22_4_jun_ds = gdal.Open(folder_data+'20150604_T22.tif', gdal.GA_ReadOnly)
    T22_4_jun_ds = T22_4_jun_ds.ReadAsArray()
    T22_4_jun = np.zeros((9500,9500,1))
    for b in range(0,T22_4_jun_ds.shape[0]):
        T22_4_jun[b, :, :] = T22_4_jun_ds[b,:].reshape(-1,1)
  
    T22_15_jun_ds = gdal.Open(folder_data+'20150615_T22.tif', gdal.GA_ReadOnly)
    T22_15_jun_ds = T22_15_jun_ds.ReadAsArray()
    T22_15_jun = np.zeros((9500,9500,1))
    for b in range(0,T22_15_jun_ds.shape[0]):
        T22_15_jun[b, :, :] = T22_15_jun_ds[b,:].reshape(-1,1)  
    
    T22_26_jun_ds = gdal.Open(folder_data+'20150626_T22.tif', gdal.GA_ReadOnly)
    T22_26_jun_ds = T22_26_jun_ds.ReadAsArray()
    T22_26_jun = np.zeros((9500,9500,1))
    for b in range(0,T22_26_jun_ds.shape[0]): 
        T22_26_jun[b, :, :] = T22_26_jun_ds[b,:].reshape(-1,1)  
        
    T22_7_jul_ds = gdal.Open(folder_data+'20150707_T22.tif', gdal.GA_ReadOnly)
    T22_7_jul_ds = T22_7_jul_ds.ReadAsArray()
    T22_7_jul = np.zeros((9500,9500,1))
    for b in range(0,T22_7_jul_ds.shape[0]):
        T22_7_jul[b, :, :] = T22_7_jul_ds[b,:].reshape(-1,1) 
        
    T22_18_jul_ds = gdal.Open(folder_data+'20150718_T22.tif', gdal.GA_ReadOnly)
    T22_18_jul_ds = T22_18_jul_ds.ReadAsArray()
    T22_18_jul = np.zeros((9500,9500,1))
    for b in range(0,T22_18_jul_ds.shape[0]):
        T22_18_jul[b, :, :] = T22_18_jul_ds[b,:].reshape(-1,1)
        
    T22_29_jul_ds = gdal.Open(folder_data+'20150729_T22.tif', gdal.GA_ReadOnly)
    T22_29_jul_ds = T22_29_jul_ds.ReadAsArray()
    T22_29_jul = np.zeros((9500,9500,1))
    for b in range(0,T22_29_jul_ds.shape[0]):
        T22_29_jul[b, :, :] = T22_29_jul_ds[b,:].reshape(-1,1)
        
    T22_9_ago_ds = gdal.Open(folder_data+'20150809_T22.tif', gdal.GA_ReadOnly)
    T22_9_ago_ds = T22_9_ago_ds.ReadAsArray()
    T22_9_ago = np.zeros((9500,9500,1))
    for b in range(0,T22_9_ago_ds.shape[0]):
        T22_9_ago[b, :, :] = T22_9_ago_ds[b,:].reshape(-1,1)
  
    T22_20_ago_ds = gdal.Open(folder_data+'20150820_T22.tif', gdal.GA_ReadOnly)
    T22_20_ago_ds = T22_20_ago_ds.ReadAsArray()
    T22_20_ago = np.zeros((9500,9500,1))
    for b in range(0,T22_20_ago_ds.shape[0]):
        T22_20_ago[b, :, :] = T22_20_ago_ds[b,:].reshape(-1,1)
        
    T22_31_ago_ds = gdal.Open(folder_data+'20150831_T22.tif', gdal.GA_ReadOnly)
    T22_31_ago_ds = T22_31_ago_ds.ReadAsArray()
    T22_31_ago = np.zeros((9500,9500,1))
    for b in range(0,T22_31_ago_ds.shape[0]):
        T22_31_ago[b, :, :] = T22_31_ago_ds[b,:].reshape(-1,1)

    return T11_4_jun,T11_15_jun,T11_26_jun,T11_7_jul,T11_18_jul,T11_29_jul,T11_9_ago,T11_20_ago,T11_31_ago,T22_4_jun,T22_15_jun,T22_26_jun,T22_7_jul,T22_18_jul,T22_29_jul,T22_9_ago,T22_20_ago,T22_31_ago

def Confusion_Dataframe(y_test_n,y_pred,class_names,nombre):
    # matriz de confusion
    df = pd.DataFrame()
    df['truth'] = y_test_n
    df['predict'] = y_pred.flatten()
   
    c_matrix = pd.crosstab(df['truth'], df['predict'], margins=True)

    return c_matrix



###########################################
import os
import os.path
from osgeo import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
t_ini = time()

folder_data = '../archivos/GeoTiff_Observables/21x21/'
test_size = 'test10'

# lectura y formateo de los datos
T11_4_jun,T11_15_jun,T11_26_jun,T11_7_jul,T11_18_jul,T11_29_jul,T11_9_ago,T11_20_ago,T11_31_ago,T22_4_jun,T22_15_jun,T22_26_jun,T22_7_jul,T22_18_jul,T22_29_jul,T22_9_ago,T22_20_ago,T22_31_ago = Formateo(folder_data)

# lectura crop_type
crop_type_ds = gdal.Open('../archivos/SHP/crop_type_complete_BXII.gtif', gdal.GA_ReadOnly)
crop_type = crop_type_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
del crop_type_ds


plt.imshow(crop_type, cmap=plt.cm.Spectral)
plt.title('crop_type')
plt.show()


# muestra todas las clases y cuantos pixeles tiene cada una
classes = np.unique(crop_type)
for c in classes:
    print('Class {c} contains {n} pixels'.format(c=c, n=(crop_type == c).sum()))

n_samples = (crop_type > 0).sum()
print('\nWe have {n} samples'.format(n=n_samples))

labels = np.unique(crop_type[crop_type > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

y = crop_type[crop_type > 0]
# guardamos los datos del feature donde el crop_type sea mayor de 0 (datos utiles)

# de X...X9 son features T11, de X10...X18 son las 9 fechas del T22

X = T11_4_jun[crop_type > 0, :]
X2 = T11_15_jun[crop_type > 0, :]
X3 = T11_26_jun[crop_type > 0, :]
X4 = T11_7_jul[crop_type > 0, :]
X5 = T11_18_jul[crop_type > 0, :]
X6 = T11_29_jul[crop_type > 0, :]
X7 = T11_9_ago[crop_type > 0, :]
X8 = T11_20_ago[crop_type > 0, :]
X9 = T11_31_ago[crop_type > 0, :]

X10 = T22_4_jun[crop_type > 0, :]
X11 = T22_15_jun[crop_type > 0, :]
X12 = T22_26_jun[crop_type > 0, :]
X13 = T22_7_jul[crop_type > 0, :]
X14 = T22_18_jul[crop_type > 0, :]
X15 = T22_29_jul[crop_type > 0, :]
X16 = T22_9_ago[crop_type > 0, :]
X17 = T22_20_ago[crop_type > 0, :]
X18 = T22_31_ago[crop_type > 0, :]


# añadimos columnas de las distintas fechas
X_18fechas = np.hstack((X,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18))


print('Our X matrix is sized: {sz}'.format(sz=X_18fechas.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))

# dB
X_18fechas = 10*np.log10(X_18fechas)


from sklearn.model_selection import train_test_split
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_18fechas, y, test_size=0.1)
# test size mas alto -> mas error (menor tamaño del train size)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0, oob_score=True, max_depth=None, n_jobs=-1)
rf.fit(X_train_n, y_train_n)

# score
oob = round((rf.oob_score_ * 100),2)
print "\nOOB Score: ",round(oob,2),"%"
score = rf.score(X_train_n,y_train_n)*100
print "\nScore: ",round(score,2),"%"
y_pred = rf.predict(X_test_n)

class_names = ['Alfalfa', 'Wheat', 'Cotton', 'Corn', 'Sugar beet', 'Tomato', 'Sunflower', 'Fallow', 'Beans', 'Quinoa', 'Broccoli', 'Carrot', 'Broccoli 2nd', 'Carrot 2nd', 'Cauliflower', 'Cauliflower 2nd', 'Chickpea', 'Onion', 'Palmeras', 'Pea', 'Pea 2nd', 'Potato','Solar Farm','All']


# nombre archivo salida 
nombre = 'matrix test 10%'
c_matrix = Confusion_Dataframe(y_test_n,y_pred,class_names,nombre)

# kappa
from sklearn.metrics import cohen_kappa_score
k = cohen_kappa_score(y_test_n, y_pred.flatten())
print '\nKappa score:',k


#############################  ACCURACIES #####################
total_acc = Total_Accuracy(c_matrix)

df_acc = pd.DataFrame()
df_acc['Accuracy'] = 'k','total_acc','oob','score'
df_acc['Value'] = k,total_acc,oob,score

# ACCURACIES -> csv
my_file = '../archivos/resultados/'+test_size+'/'+'Accuracies'
if os.path.isfile(my_file):
    os.remove(my_file)
df_acc.to_csv(r'../archivos/resultados/'+test_size+'/'+'Accuracies', header=True, index=True, sep=' ', mode='a')




#############################  USER #####################

user_acc = User_Accuracy(c_matrix,class_names)
df_user = pd.DataFrame()
# insertamos en el dataframe las porcentajes de user acc
df_user['User Accuracy'] = user_acc
# insertamos las etiquetas
df_user['Class'] = class_names[0:23]
print df_user

# User_Accuracy -> csv
my_file = '../archivos/resultados/'+test_size+'/'+'User Accuracy'
if os.path.isfile(my_file):
    os.remove(my_file)
df_user.to_csv(r'../archivos/resultados/'+test_size+'/'+'User Accuracy', header=True, index=True, sep=' ', mode='a')



#############################  PRODUCER #####################

prod_acc = Producer_Accuracy(c_matrix,class_names)
df_prod = pd.DataFrame()
df_prod['Producer Accuracy'] = prod_acc
df_prod['Class'] = class_names[0:23]
print df_prod

# Producer_Accuracy -> csv
my_file = '../archivos/resultados/'+test_size+'/'+'Producer Accuracy'
if os.path.isfile(my_file):
    os.remove(my_file)
df_prod.to_csv(r'../archivos/resultados/'+test_size+'/'+'Producer Accuracy', header=True, index=True, sep=' ', mode='a')



#############################  CONFUSION #####################

print "\nMatriz de confusion \n--------------------"
c_matrix.columns = class_names
c_matrix.index = class_names
print c_matrix

# confusion matrix -> csv
my_file = '../archivos/resultados/'+test_size+'/'+nombre
if os.path.isfile(my_file):
    os.remove(my_file)
c_matrix.to_csv(r'../archivos/resultados/'+test_size+'/'+nombre, header=True, index=True, sep=' ', mode='a')



#############################  MAPAS #####################


clf2 = rf.predict(X_18fechas)

resul = np.zeros((9500,9500))

# construimos la imagen resultado
indx, indy = (crop_type > 0).nonzero()
# insertamos en esos indices los valores predichos
resul[indx[0:len(indx)],indy[0:len(indx)]] = clf2



plt.imshow(crop_type, cmap=plt.cm.Spectral)
plt.title('crop_type_original recortado')
plt.savefig('crop_type_original recortado.png')
plt.show()

plt.imshow(resul, cmap=plt.cm.Spectral)
plt.title('clases resultantes')
plt.savefig('clases resultantes.png')
plt.show()

#####  Diferencia
dif = np.zeros((9500,9500))

# indices donde hay diferencias
indx2, indy2 = (crop_type != resul).nonzero()
# insertamos un 1 en esos indices
dif[indx2[0:len(indx2)],indy2[0:len(indx2)]]=1

plt.imshow(dif, cmap=plt.cm.Spectral)
plt.title('diferencias')
plt.show()




#Evolucion temporal: para cada fecha obtener todos los pixeles de cada cultivo y hallar media y desv estandar
############################# MEDIAS T11 #####################

# MEDIA T11
X_T11 = np.zeros((9,9500,9500))
X_T11[0,:,:] = T11_4_jun.reshape(9500,9500)
X_T11[1,:,:] = T11_15_jun.reshape(9500,9500)
X_T11[2,:,:] = T11_26_jun.reshape(9500,9500)
X_T11[3,:,:] = T11_7_jul.reshape(9500,9500)
X_T11[4,:,:] = T11_18_jul.reshape(9500,9500)
X_T11[5,:,:] = T11_29_jul.reshape(9500,9500)
X_T11[6,:,:] = T11_9_ago.reshape(9500,9500)
X_T11[7,:,:] = T11_20_ago.reshape(9500,9500)
X_T11[8,:,:] = T11_31_ago.reshape(9500,9500)

X_T11 = 10*np.log10(X_T11)


mediaX_T11 = np.zeros((9,24))
desvX_T11 = np.zeros((9,24))
# imagenes
for i in range(0,9):
    # features
    for clase in range (1,24):
        # indices de pixeles de la clase 1
        pixx,pixy = (crop_type==clase).nonzero()
        # guardamos para cada feature e imagen su media
        mediaX_T11[i,clase] = np.mean(X_T11[i,pixx,pixy])
        # desv de cada clase en cada fecha
        desvX_T11[i,clase] = np.std(X_T11[i,pixx,pixy])
# plot
x_axis = np.arange(1,10,1)

for i in range(1,24):
    fig = plt.figure(1)
    plt.errorbar(x_axis, mediaX_T11[:,i], desvX_T11[:,i], marker='o')
    plt.title("Evolucion temporal T11 " +class_names[i-1])
    
    # Y axes
    plt.ylim(-10,1)
    plt.show()

    fig.savefig('../archivos/resultados/'+test_size+'/'+'T11 '+class_names[i-1],bbox_inches='tight', pad_inches=0)

######################### plot misma grafica
fig = plt.figure(1)
ax = fig.add_subplot(111)
dev = np.std(mediaX_T11[:,i])
for i in range(1,23):
    plt.plot(x_axis, mediaX_T11[:,i], label="T11 "+class_names[i])
    
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
fig.savefig('../archivos/resultados/'+test_size+'/'+'T11',bbox_inches='tight', pad_inches=0)


############################# MEDIAS T22 #####################

# MEDIA T22
X_T22 = np.zeros((9,9500,9500))
X_T22[0,:,:] = T22_4_jun.reshape(9500,9500)
X_T22[1,:,:] = T22_15_jun.reshape(9500,9500)
X_T22[2,:,:] = T22_26_jun.reshape(9500,9500)
X_T22[3,:,:] = T22_7_jul.reshape(9500,9500)
X_T22[4,:,:] = T22_18_jul.reshape(9500,9500)
X_T22[5,:,:] = T22_29_jul.reshape(9500,9500)
X_T22[6,:,:] = T22_9_ago.reshape(9500,9500)
X_T22[7,:,:] = T22_20_ago.reshape(9500,9500)
X_T22[8,:,:] = T22_31_ago.reshape(9500,9500)

X_T22 = 10*np.log10(X_T22)


mediaX_T22 = np.zeros((9,24))
desvX_T22 = np.zeros((9,24))
# imagenes
for i in range(0,9):
    # features
    for clase in range (1,24):
        # indices de pixeles de la clase 1
        pixx,pixy = (crop_type==clase).nonzero()
        # guardamos para cada feature e imagen su media
        mediaX_T22[i,clase] = np.mean(X_T22[i,pixx,pixy])
        # la 1a columna de mediaX_T22 es 0 para que los indices sean mas comodos
        desvX_T22[i,clase] = np.std(X_T22[i,pixx,pixy])

# plot
for i in range(1,24):
    fig = plt.figure(1)
    
    
    x_axis = np.arange(1,10,1)
    plt.errorbar(x_axis, mediaX_T22[:,i], desvX_T22[:,i], marker='o')
    plt.title("Evolucion temporal T22 " +class_names[i-1])
    
    # Y axes
    plt.ylim(-19,-6)
    plt.show()
    
    fig.savefig('../archivos/resultados/'+test_size+'/'+'T22 '+class_names[i-1],bbox_inches='tight', pad_inches=0)


    
    

######################### plot misma grafica
fig = plt.figure(1)
ax = fig.add_subplot(111)
dev = np.std(mediaX_T22[:,i])
for i in range(1,23):
    plt.plot(x_axis, mediaX_T22[:,i], label="T22 "+class_names[i])
    
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
fig.savefig('../archivos/resultados/'+test_size+'/'+'T22',bbox_inches='tight', pad_inches=0)

t_fin = time()
print '\nTiempo de ejecucion en minutos:',round((t_fin-t_ini)/60,2)