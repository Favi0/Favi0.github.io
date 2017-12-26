Hola, en este tutorial voy a tratar de explicaros de forma resumida, en que consiste el algoritmo de **regresión logística**.


Notación 
-------------

>  - **m=número total de muestras**
>  -  **n =número total de características por muestra**
>  - **w='*pesos*' o '*weights*'**, son parámetros del modelo
>  - **X =vector de muestras**
>  -  **Y =vector de 'clases'**

![X](http://latex.codecogs.com/gif.latex?X%3D%5Cbegin%7Bbmatrix%7D%20x_1%5E%7B%281%29%7D%20x_1%5E%7B%282%29%7D...x_1%5E%7B%28m%29%7D%20%5C%5Cx_2%5E%7B%281%29%7D%20x_2%5E%7B%282%29%7D...x_2%5E%7B%28m%29%7D%5C%5C...%5C%5C%20x_n%5E%7B%281%29%7D%20x_n%5E%7B%282%29%7D...x_n%5E%7B%28m%29%7D%20%5Cend%7Bbmatrix%7D%2C%20Y%3D%5Cbegin%7Bbmatrix%7D%20y%5E%7B%281%29%7D%20y%5E%7B%282%29%7D...y%5E%7B%28m%29%7D%20%5Cend%7Bbmatrix%7D)

Entonces X es la concatenación de cada muestra individual Xi. A su vez cada muestra tendrá asociada una salida yi que indicará si pertenece (1) o no (0) a la clase que estemos clasificando.

![enter image description here](http://latex.codecogs.com/gif.latex?x%5E%7B%28i%29%7D%3D%5Cbegin%7Bbmatrix%7D%20x_1%5E%7B%28i%29%7D%20%5C%5C%20x_2%5E%7B%28i%29%7D%20%5C%5C...%5C%5Cx_n%5E%7B%28i%29%7D%20%5Cend%7Bbmatrix%7D%20%2C%20y%5E%7B%28i%29%7D%3D%5Cbegin%7Bbmatrix%7D%20y%5E%7B%28i%29%7D%20%5Cend%7Bbmatrix%7D)

 
**La dimensión de W dependerá de n**

![enter image description here](http://latex.codecogs.com/gif.latex?W%3D%5Cbegin%7Bbmatrix%7Dw_1%20%5C%5C%20w_2%5C%5C...%5C%5C%20w_n%20%5Cend%7Bbmatrix%7D%2CW%5E%7Bt%7D%3D%5Cbegin%7Bbmatrix%7Dw_1%20%5C%20w_2...%5C%20w_n%20%5Cend%7Bbmatrix%7D)

Introducción
-------------

Desde el punto de vista del aprendizaje automático, la regresión logística es un modelo de clasificación fácil de implementar y con buen desempeño en problemas con clases linealmente separables. Podemos resumir algunas de sus características:


> **Regresión Logística:**
> 
> - Es un modelo probabilístico.
> - Buen desempeño en problemas de clasificación binaria.
> - Puede utilizarse en problemas de clasificación multiclase (uno contra todos).
> - Consta de parámetros que deberán aprenderse (pesos).
> - Hace uso de una función de Coste y un optimizador para la misma.

Desarrollo del Modelo: Intuición 
-------------------

La regresión logistica es un modelo basado en la probabilidad, es decir que el modelo nos devolverá las probabilidades que tienen las distintas instancias de pertenecer a alguna de clases con las que estemos trabajando. Para entender mejor la idea, veamos unos mínimos conceptos previos.

La probabilidad ***p*** es la medida de estimación que tenemos de que ocurra un evento determinado, ej. que una muestra que queremos clasificar pertenezca a una de las clases y=0 o y=1. 
**p tomará valores comprendidos entre 0 (0%) y 1 (100%).** 

Llamemos **Razón de probabilidades ** (conocida popularmente como **odds Ratio**) a la razón de *la probabilidad de que suceda ese evento*, sobre, la *probabilidad de que no suceda ese evento*. Es decir,

![enter image description here](http://latex.codecogs.com/gif.latex?OR=%5Cfrac%7Bp%7D%7B%281-p%29%7D)

Podemos definir entonces la función logit como:

![logit](http://latex.codecogs.com/gif.latex?logit%28p%29=%5Clog%20%5Cfrac%7Bp%7D%7B%281-p%29%7D)

Puede verse que es aplicar logaritmo sobre la expresión anterior. Esta función tomará como entrada valores comprendidos entre 0 y 1, y devolverá números comprendidos en todo el rango real, el cual podemos usar para expresar una relación lineal

![enter image description here](http://latex.codecogs.com/gif.latex?logit%28p%28y=1%7Cx%29%29=w_0%20x_1%20&plus;w_1x_1&plus;...&plus;w_nx_n=%5Csum_%7Bi=0%7D%5E%7Bn%7D%20w_ix_i=w%5E%7BT%7Dx)


![zeta](http://latex.codecogs.com/gif.latex?z=w_1x_1&plus;...&plus;w_nx_n=w%5Etx)

donde p(y=1|x) es la *probabilidad condicional* de que una muestra x pertenezca a la clase 1, dado su determinado vector de caracteristicas x1,..,xn. 
Notar que esta combinación lineal puede expresarse como un producto de matrices por medio de la transpuesta (dado que todas las muestras tienen la misma dimensión, o el mismo número de características) en vez de estar haciendo operaciones por cada elemento de la muestra.

> Como contamos con un conjunto de muestras X, compuestas por n características cada una, y los parámetros W del modelo..**¿no habría alguna forma de obtener una probabilidad en la salida **(es decir una predicción de la clase a la que pertenece la muestra)**, que es lo que realmente nos interesa?**

Esto se obtiene con la función inversa a *logit* , conocida popularmente como función *Sigmoidea* o función logística. Esta función tomará números reales en la entrada y devolverá probabilidades entre 0 y 1 en la salida.

![enter image description here](http://latex.codecogs.com/gif.latex?P%28y=1%7Cx%29=%5Csigma%20%28z%29=%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-z%7D%7D)

Donde z será la combinación lineal descripta previamente. 

> Esta función es conocida como activación , hipótesis o no linearidad, sobre todo en redes neuronales.

![sigmoid](https://k60.kn3.net/B/A/8/5/C/8/8CB.png)

De la gráfica puede verse que rapidamente converge a 1 o 0, en muchas aplicaciones donde solo nos importa clasificar la clase correctamente y no la probabilidad podemos usar un valor determinado como umbral 

![boundary](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D=%20%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Blcc%7D%201%20&%20si%20&%20%5Csigma%28z%29%5Cgeq%200.5%20%5C%5C%20%5C%5C%200%20&%20si%20&%20%5Csigma%28z%29%3C0.5%20%5Cend%7Barray%7D%20%5Cright.)

Hasta aquí llega la parte de la propagación hacia delante del algoritmo, la función sigmoid nos dará una predicción para cada instancia, las cuales luego compararemos con el label **y** de esas muestra en la etapa de entrenamiento, para determinar el error de la predicción y actualizar los parámetros del modelo.

Podrían utilizarse otras funciones en vez de la *sigmoidea* , pero esta presenta ciertas ventajas que nos serán útiles mas adelante:

 - Se relaciona de forma sencilla las posibilidades/chances de sucesos  descriptas al inicio.
 - Devuelve una probabilidad de pertenecer a la clase.
 - Los gradientes(derivadas) son fáciles de calcular.

## Función de Coste ##

Una consideración quizás no tan obvia a esta altura es que los **pesos** del modelo que entrenamos son los responsables de predecir de forma precisa la clase de las instancias del ***test set*** (instancias nuevas nunca vistas por el modelo, que queremos clasificar).

Usamos una función de Coste ***J***  para "medir" que tan cerca están los valores que predecimos para cada una de nuestras instancias del ***test set***, a los de sus valores reales o los que entrenamos el modelo usando el ***training set***. Mientras mas cercanos mejor será el modelo.

> *No tendremos en cuenta el desarrollo matemático para llegar a la expresión de J.*

La función de Coste para la regresión logística (conocida como ***Logistic Cost***) es 


 ![Coste logistico](http://latex.codecogs.com/gif.latex?J%28w%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5By%20%5Clog%20%28%5Csigma%20%28z%29%29&plus;%281-y%29%20%5Clog%20%281-%5Csigma%20%28z%29%29%5D)

La cual puede reescribirse para cada muestra como

![enter image description here](http://latex.codecogs.com/gif.latex?J%28%5Csigma%20%28z%29,y%29%29=%20%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Blcc%7D%20-%5Clog%20%28%5Csigma%20%28z%29%29%20&%20si%20&%20y=%201%20%5C%5C%20-%5Clog%20%281-%5Csigma%20%28z%29%29%20&%20si%20&%20y=%200%20%5C%5C%20%5Cend%7Barray%7D%20%5Cright.)

![cost](https://k60.kn3.net/A/C/2/3/F/D/DDE.png)

Analizando esta ecuación se ve que el primer término se vuelve cero para y=0 , de igual manera que el segundo término se vuelve cero para y=1. Tambien se deduce que si nuestra predicción es errónea el coste se va para el infinito (*se penaliza predicciones incorrectas con un coste elevado*).

> Una de las razones del uso de esta función de coste es que es una
> ***función convexa*** y nos asegurará la convergencia a un mínimo global.

## Aprendizaje de los pesos ##

Debemos **optimizar** esta función de coste J durante el proceso de entrenamiento del modelo, lo cual logramos minizando J. 
Como dijimos que la función de coste logística es **convexa**, podremos utilizar un algoritmo de optimización que ádemas de ser simple es bastante poderoso, conocido popularmente como ***Gradient Descent*** el cual nos permitirá encontrar los pesos que minimizan J para clasificar las muestras en nuestro **training set**.

La idea de Gradient Descent, tal como indica su nombre, es que vamos "descendiendo" por la función hasta llegar a un mínimo local o global. En cada iteración del algoritmo, tomamos un paso en la dirección del gradiente determinado tanto por la magnitud del gradiente como de nuestra tasa de aprendizaje o **Learning rate** (Hiperparámetro de GD).

 FOTO
 
 Usando GD, podemos actualizar los pesos Wi a travéz de alejarnos un paso del gradiente $\triangledown J$ de la función J(w), es decir

![Update Rule](http://latex.codecogs.com/gif.latex?w:=w&plus;%5CDelta%20w)  

Donde la variación del peso se define como el gradiente negativo multiplicado por la tasa de aprendizaje 

![variacion](http://latex.codecogs.com/gif.latex?%5CDelta%20w=-%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_%7Bi%7D%7D)

Para calcular el gradiente de J, tendremos que calcular la derivada parcial de J **con respecto a cada peso** *wi*.

Se puede demostrar de forma sencilla derivando con respecto a w que este gradiente tomará la siguiente forma (tambien habrá que derivar la función sigmoidea con respecto a w)

![derivada](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20J%28w%29%7D%7B%5Cpartial%20w_%7Bi%7D%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28%5Csigma%20%28z%5Ei%29-y%5Ei%29%29x%5E%7Bi%7D)

entonces obtenemos 

![enter image description here](http://latex.codecogs.com/gif.latex?w%3A%3Dw-%5Cfrac%7B%5Calpha%20%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28%5Csigma%20%28z%5Ei%29-y%5Ei%29%29x%5E%7Bi%7D)

Entonces en cada pasada del algoritmo deberemos actualizar los pesos en ese factor.

## Algoritmo##

Lo explicado previamente puede ser expresado en una versión vectorizada, pudiendo ahorrarnos el tener que operar sobre cada muestra xi de forma individual, lo que nos permitirá eliminar un lazo de 1 a m sobre cada iteración o época del algoritmo , es decir pasar de O(n^2) a O(n)  

```python
#----PSEUDOCÓDIGO  ALGORITMO VECTORIZADO----

#ENTRENAMIENTO:
#sobre el training set, X=X_train
for(i=0;i+1;iteraciones){

Z  = W.T*X           #producto de matrices
A  = sigmoid(Z)
J  = -1/m*sum(Y*log(A)+(1-Y)*log(1-A))
dw = 1/m*(X*(A-Y).T) #producto de matrices

w  = w-learning_rate*dw
mostrarCosto(J)      #vemos como cambia J en cada it
}

#PREDICCIONES:
#sobre el test set, X=X_test
A  = sigmoid(W.T*X)
if (A[:,i] > 0.5):
            Y_predict[:, i] = 1
            
elif (A[:,i] <= 0.5):
            Y_predict[:, i] = 0

```
