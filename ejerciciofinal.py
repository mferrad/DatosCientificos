"""
==================
Final Assignment
==================

Entre 10 y 11 minutos
Fs = 512


|---- BASELINE --------|
|---- TOSER ------|
|---- RESPIRAR FONDO ------- |
|---- RESPIRAR RAPIDO ----|
|---- CUENTA MENTAL --------|
|---- COLORES VIOLETA ------|
|---- COLORES ROJO --------|
|---- SONREIR -----|
|---- DESEGRADABLE -----| 
|---- AGRADABLE --------|
|---- PESTANEOS CODIGO ------ |

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por 
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Toser: Probablemente queden registrados como picos, provocados por el propio movimiento de la cabeza.
* Respirar fondo vs respirar rápido: quizás puede haber un cambio en alguna frecuencia.
* Cuenta mental: Está reportado que esto genera cambios en las frecuencias altas gamma y beta, de entre 20-30 Hz.
* Colores violeta / rojo:  de acá primero pueden intentar ver si hay cambio en relación a baseline en la frecuencia
de 10 Hz porque para ambos casos cerré los ojos.  Luego pueden intentar ver si un clasificador les puede diferenciar las clases.
* Sonreir: esto quizás genere algunos artefactos, picos en la señal debido al movimiento de la cara.
* Agradable/Desagradable: aca no tengo idea, prueben contra baseline.  No hay nada reportado así.
* Pestañeos:  En esta parte hay pestañeos que pueden intentar extraer.


Los datos, el registro de EEG y el video, están disponibles en el siguiente link:
https://drive.google.com/file/d/1ByQDK4ZPxbqw7T17k--avTcgSCCzs3vi/view?usp=sharing

"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.cluster import KMeans

import math
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord
from scipy.signal import butter, lfilter
from sklearn import preprocessing

# El protocolo experimental que implementamos tiene 2 datasets:
# 1- Dataset de las señales de EEG
# 2- El video de las imágenes.
#
#
# La idea es tomar estos datasets y derivar de forma automática las diferentes secciones.  Esto se puede hacer en base self-supervised, es
# decir tomar los datos de algún dataset, derivar los labels para cada secciones y luego intentar implementar un clasificador multiclase.
#
# Tienen que entregar un PDF, tipo Markdown con código, gráficos y cualquier insight obtenido del dataset.

signals = pd.read_csv('data/B001/eeg.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Estructura de la informacion:')
print(signals.head())

data = signals.values

## Reconstruyo la señal original

i=0
limit=len(data[:,1])


while i<limit-1:
    
    if ((data[i+1,1]-data[i,1])>1):
        # print(f'fix {data[i,1]} and {data[i+1,1]}')       
        newrow=[(data[i,0]+data[i+1,0])/2,data[i,1]+1,(data[i,2]+data[i+1,2])/2,(data[i,3]+data[i+1,3])/2,(data[i,4]+data[i+1,4])/2,(data[i,5]+data[i+1,5])/2]
        data=np.insert(data, i+1, newrow, 0)
        limit=limit+1

    elif ((data[i+1,1]-data[i,1])<0):
        if(data[i+1,1]==0 and data[i,1]==99):
            i=i+1
            continue
        else:
            if( data[i,1]==99):
                # print(f'fix2 {data[i,1]} and {data[i+1,1]}')                
                newrow=[(data[i,0]+data[i+1,0])/2,0,(data[i,2]+data[i+1,2])/2,(data[i,3]+data[i+1,3])/2,(data[i,4]+data[i+1,4])/2,(data[i,5]+data[i+1,5])/2]
                data=np.insert(data, i+1, newrow, 0)
                limit=limit+1
                

            else:                
                #print(f'fix3 {data[i,1]} and {data[i+1,1]}')                
                newrow=[(data[i,0]+data[i+1,0])/2,data[i,1]+1,(data[i,2]+data[i+1,2])/2,(data[i,3]+data[i+1,3])/2,(data[i,4]+data[i+1,4])/2,(data[i,5]+data[i+1,5])/2]
                data=np.insert(data, i+1, newrow, 0)
                limit=limit+1
               
    
    i=i+1

## Creo una columna adicional con una variable tiempo precisa

data=np.insert(data,6, range(limit), 1)
data[:,6]=data[:,6]/512

## Funciones de generacion de features

def crest_factor(x):
    return np.max(np.abs(x))/np.sqrt(np.mean(np.square(x)))

def hjorth(a):
    r"""
    Compute Hjorth parameters [HJO70]_.
    .. math::
        Activity = m_0 = \sigma_{a}^2
    .. math::
        Complexity = m_2 = \sigma_{d}/ \sigma_{a}
    .. math::
        Morbidity = m_4 =  \frac{\sigma_{dd}/ \sigma_{d}}{m_2}
    Where:
    :math:`\sigma_{x}^2` is the mean power of a signal :math:`x`. That is, its variance, if it's mean is zero.
    :math:`a`, :math:`d` and :math:`dd` represent the original signal, its first and second derivatives, respectively.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appear to uses a non normalised (by the length of the signal) definition of the activity:
        .. math::
            \sigma_{a}^2 = \sum{\mathbf{x}[i]^2}
        As opposed to
        .. math::
            \sigma_{a}^2 = \frac{1}{n}\sum{\mathbf{x}[i]^2}
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: activity, complexity and morbidity
    :rtype: tuple(float, float, float)
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> activity, complexity, morbidity = pr.univariate.hjorth(noise)
    """

    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity

def pfd(a):
    r"""
    Compute Petrosian Fractal Dimension of a time series [PET95]_.
    It is defined by:
    .. math::
        \frac{log(N)}{log(N) + log(\frac{N}{N+0.4N_{\delta}})}
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which implemented an apparently erroneous formulae:
        .. math::
            \frac{log(N)}{log(N) + log(\frac{N}{N}+0.4N_{\delta})}
    Where:
    :math:`N` is the length of the time series, and
    :math:`N_{\delta}` is the number of sign changes.
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: the Petrosian Fractal Dimension; a scalar.
    :rtype: float
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> pr.univariate.pdf(noise)
    """

    diff = np.diff(a)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]

    # Number of sign changes in derivative of the signal
    N_delta = np.sum(prod < 0)
    n = len(a)

    return np.log(n)/(np.log(n)+np.log(n/(n+0.4*N_delta)))





def MyPlot(minutos,segundos,ventana,data):

    posicionInicial=round((minutos*60+segundos)*512)
    posicionFinal=round((minutos*60+segundos+ventana)*512)
    eeg = data[posicionInicial:posicionFinal,2]
    eeg = eeg-getBaseline(eeg)
    tiempo= data[posicionInicial:posicionFinal,6]

    plt.plot(tiempo,eeg,'r', label='EEG')
    plt.xlabel('t');
    plt.ylabel('eeg(t)');
    plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
    plt.show()

from scipy.fftpack import fft

def MyFrecPlot(minutos,segundos,ventana,data):
    posicionInicial=round((minutos*60+segundos)*512)
    posicionFinal=round((minutos*60+segundos+ventana)*512)
    eeg = data[posicionInicial:posicionFinal,2]
    eeg = eeg-getBaseline(eeg)

    sr = 512.0
    X = fft(eeg)
    N = len(X)
    n = np.arange(N)
    # get the sampling rate
    T = N/sr
    freq = n/T 
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    plt.figure(figsize = (12, 6))
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()

from scipy.interpolate import interp1d

def getBaseline(eeg):

    # Get 100 points from 0 .. len(eeg)  [0,123,340,...,len(eeg)]
    x = range(0,len(eeg),100)
    y = eeg[x]                                                  # Get the signal values on those points
    f = interp1d(x, y,fill_value="extrapolate")                 # Estimate a function that will interpolate those points. 
                                                            # This will estimate a waveform based solely on those points.   
    f2 = interp1d(x, y, kind='cubic',fill_value="extrapolate")  # Replicate the same with a cubic function

    # Now regenerate the signal based on the estimated function 'f' and get a signal from that.
    baseline = f(range(len(eeg)))

    # Finally, substract those points from the original signal.
    return(baseline)


def getFrecFetures(minutos,segundos,ventana,data):

    posicionInicial=round((minutos*60+segundos)*512)
    posicionFinal=round((minutos*60+segundos+ventana)*512)
    eeg = data[posicionInicial:posicionFinal,2]
    eeg = eeg-getBaseline(eeg)
    sr = 512.0
    X = fft(eeg)
    N = len(X)
    n = np.arange(N)
    # get the sampling rate
    T = N/sr
    freq = n/T 
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    v_oneside=np.abs(X[:n_oneside])

    f=1
    margen=0.5
    limite=25
    incremento=1

    features=[]

    while (f<=limite):
        indexes=np.where(np.logical_and(f_oneside>(f-margen),f_oneside<(f+margen)))
        mean=np.mean(v_oneside[indexes])
        f=f+incremento
        features.insert(1,mean)

    f=27.5
    margen=2.5
    limite=50
    incremento=5

    while (f<=limite):
        indexes=np.where(np.logical_and(f_oneside>(f-margen),f_oneside<(f+margen)))
        mean=np.mean(v_oneside[indexes])
        f=f+incremento
        features.insert(1,mean)

    f=55
    margen=5
    limite=100
    incremento=10

    while (f<=limite):
        indexes=np.where(np.logical_and(f_oneside>(f-margen),f_oneside<(f+margen)))
        mean=np.mean(v_oneside[indexes])
        f=f+incremento
        features.insert(1,mean)

    return(features)



def getTimeFetures(minutos,segundos,ventana,data):

    features=[]
    posicionInicial=round((minutos*60+segundos)*512)
    posicionFinal=round((minutos*60+segundos+ventana)*512)
    eeg = data[posicionInicial:posicionFinal,2]
    eeg = eeg-getBaseline(eeg)

    ptp = abs(np.max(eeg)) + abs(np.min(eeg))
    features.insert(1,ptp)

    rms = np.sqrt(np.mean(eeg**2))
    features.insert(1,rms)

    cf = crest_factor(eeg)
    features.insert(1,cf)

    from collections import Counter
    from scipy import stats

    entropy = stats.entropy(list(Counter(eeg).values()), base=2)
    features.insert(1,entropy)

    activity, complexity, morbidity = hjorth(eeg)
    features.insert(1,activity)
    features.insert(1,complexity)
    features.insert(1,morbidity)
    fractal = pfd(eeg)
    features.insert(1,fractal)

    return(features)


'''

*** Pestaneos anteriores ***

minutos=0
segundos=0
ventana=10

|---- BASELINE --------|

minutos=0
segundos=9
ventana=55

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)


*** Pestaneos anteriores ***

minutos=0
segundos=65
ventana=4

|---- TOSER ------|

minutos=0
segundos=69
ventana=60

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)


*** Pestaneos anteriores ***

minutos=0
segundos=130
ventana=5

|---- RESPIRAR FONDO ------- |

minutos=0
segundos=135
ventana=70

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)

*** Pestaneos anteriores ***

minutos=0
segundos=210
ventana=3

|---- RESPIRAR RAPIDO ----|

minutos=0
segundos=213
ventana=57

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)


*** Pestaneos anteriores ***

minutos=0
segundos=270
ventana=6

|---- CUENTA MENTAL --------|

minutos=0
segundos=270
ventana=43

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)

*** Pestaneos anteriores ***

minutos=0
segundos=313
ventana=4

|---- COLORES VIOLETA ------|

minutos=0
segundos=317
ventana=58

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)

*** Pestaneos anteriores ***

minutos=0
segundos=375
ventana=5

|---- COLORES ROJO --------|

minutos=0
segundos=380
ventana=55

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)

*** Pestaneos anteriores ***

minutos=0
segundos=435
ventana=3

|---- SONREIR -----|

minutos=0
segundos=438
ventana=60

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)

*** Pestaneos anteriores ***

minutos=0
segundos=498
ventana=3

|---- DESEGRADABLE -----| 

minutos=0
segundos=501
ventana=59

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)


*** No pestanea ***

|---- AGRADABLE --------|

minutos=0
segundos=560
ventana=58

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)


*** Pestaneos anteriores ***

minutos=0
segundos=618
ventana=4


|---- PESTANEOS CODIGO ------ |

minutos=0
segundos=622
ventana=45

MyPlot(minutos,segundos,ventana,data)
MyFrecPlot(minutos,segundos,ventana,data)

'''

# recorro la informcion util y creo features con operaciones temporales y frecuenciales, results tiene las features y  labeles el tag de a que info corresponde

anchoVentana=3
minutos=0
results= []
labels=[]

listaTags=[['BASELINE'],['TOSER'],['RESPIRAR_FONDO'],['RESPIRAR_RAPIDO'],['CUENTA_MENTAL'],['COLORES_VIOLETA'],['COLORES_ROJO'],['SONREIR'],['DESEGRADABLE'],['AGRADABLE'],['PESTANEOS_CODIGO']]
listaSegundos=[9,69,135,213,270,317,380,438,501,560,622]
listaVentanas=[55,60,70,57,43,58,55,60,59,58,45]


for j in range(len(listaTags)):
        
    segundos=listaSegundos[j]
    ventana=listaVentanas[j]
    tag=listaTags[j]
    i=1

    while(ventana> (anchoVentana*i)):
        segundos2=segundos+anchoVentana*(i-1)
        ffeatures=getFrecFetures(minutos,segundos2,anchoVentana,data)
        tfeatures=getTimeFetures(minutos,segundos2,anchoVentana,data)
        labels.append(tag[0])
        row=ffeatures+tfeatures
        results.append(row)
        i=i+1

#convierto a un np array
np_results=np.array(results)

#normalizo los resultados
norm_np_results=preprocessing.StandardScaler().fit(np_results).transform(np_results.astype(float))

#Hago un algoritmo de clustering, grafico como mejora a medida que k aumenta y calculo los grupos para un k=11

Sum_of_squared_distances = []
K = range(1,15)

for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(norm_np_results)
    Sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()

num_clusters=11
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(norm_np_results)

labelsLearnt = kmeans.labels_

resultsDf=pd.DataFrame({'LabelOrig':labels,'labelLearnt':labelsLearnt, 'count': ([1] *len(labels)) })

resultsDf=resultsDf.groupby(['labelLearnt','LabelOrig'],as_index=False).sum().sort_values(by=['labelLearnt','count'])

print(resultsDf)