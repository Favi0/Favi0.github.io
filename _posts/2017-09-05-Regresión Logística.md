Hola, en este tutorial  voy a tratar de explicaros de forma resumida, en que consiste el algoritmo de **regresión logística**.


Motivación 
-------------

Introducción
-------------

Desde el punto de vista del aprendizaje automático, la regresión logistica es un modelo de clasificación fácil de implementar y con buen desempeño en problemas con clases linealmente separables. Podemos resumir algunas de sus características:


> **Regresión Lineal:**
> 
> - Es un modelo probabilístico.
> - Buen desempeño en problemas de clasificación binaria.
> - Puede utilizarse en problemas de clasificación multiclase (uno contra todos).
> - Consta de parámetros que deberán aprenderse.
> - Hace uso de una función de Coste y un optimizador para la misma.

Desarrollo del Modelo: Intuición 
-------------------

La regresión logistica es un modelo basado en la probabilidad, es decir que el modelo nos devolverá las probabilidades que tienen las distintas instancias de pertenecer a alguna de clases con las que estemos trabajando. Para entender mejor la idea, veamos unos mínimos conceptos previos.

La probabilidad ***p*** es la medida de estimación que tenemos de que ocurra un evento determinado, ej. que una muestra que queremos clasificar pertenezca a una de las clases y=0 o y=1. 
**p tomará valores comprendidos entre 0 (0%) y 1 (100%).** 

Llamemos **odds ratio** (no hay una única traducción definida) a la razón de *la probabilidad de que suceda ese evento*, sobre, la *probabilidad de que no suceda ese evento*. Es decir,

$$\frac{p}{(1-p)}
$$

Podemos definir entonces la función logit como:
$$
logit(p)=\log \frac{p}{(1-p)}
$$

Puede verse que es aplicar logaritmo sobre la expresión anterior. Esta función tomará como entrada valores comprendidos entre 0 y 1, y devolverá números comprendidos en todo el rango real. 

Un modelo logarítmico como este puede expresarse como la combinación lineal de los parámetros del modelo *wi* (llamados *pesos* o *weights*)  y las caract de la muestra, es decir:

$$
logit(p(y=1|x))=w_0 x_1 +w_1x_1+...+w_nx_n=\sum_{i=0}^{n} w_ix_i=w^{T}x
$$

donde p(y=1|x) es la *probabilidad condicional* de que una muestra x pertenezca a la clase 1, dado su determinado vector de caracteristicas x1,..,xn. 
Notar que esta combinación lineal puede expresarse como un producto de matrices por medio de la transpuesta (dado que todas las muestras xi tienen la misma dimensión, o el mismo número de características).

> Como contamos con un conjunto de muestras X, compuestas por n características cada una, y los parámetros W del modelo..**¿no habría alguna forma de obtener una probabilidad en la salida, que es lo que realmente nos interesa?**

Esto se obtiene con la función inversa a *logit* , conocida popularmente como función *Sigmoidea* o función logística. Esta función tomará números reales en la entrada y devolverá probabilidades entre 0 y 1 en la salida.

![enter image description here](http://latex.codecogs.com/gif.latex?P%28y=1%7Cx%29=%5Csigma%20%28z%29=%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-z%7D%7D)

Donde z será la combinación lineal descripta previamente. 

> Esta función es conocida como activación , hipótesis o no linearidad, sobre todo en redes neuronales.

foto

Podrían utilizarse otras funciones en vez de la *sigmoidea* , pero esta presenta ciertas ventajas que nos serán útiles mas adelante:

 - Se relaciona de forma sencilla las posibilidades/chances de sucesos (**odds**) descriptas al inicio.
 - Los gradientes(derivadas) son fáciles de calcular.

## Función de Coste ##

Una consideración quizás no tan obvia a esta altura es que los **pesos** del modelo que entrenamos son los responsables de predecir de forma precisa los nuevos valores del ***test set*** (instancias nuevas nunca vistas por el modelo, que queremos clasificar).

Usamos una función de Coste ***J***  para "medir" que tan cerca están los valores que predecimos para cada una de nuestras instancias del ***test set***, a los de sus valores reales o los que entrenamos el modelo usando el ***training set***. Mientras mas cercanos mejor será el modelo.

> *No tendremos en cuenta el desarrollo matemático para llegar a la expresión de J.*

La función de Coste para la regresión logística (conocida como ***Logistic Cost***) es 


 ![Coste logistico](http://latex.codecogs.com/gif.latex?J%28w%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5By%20%5Clog%20%28%5Csigma%20%28z%29%29&plus;%281-y%29%20%5Clog%20%281-%5Csigma%20%28z%29%29%5D)

foto

Analizando esta ecuación se ve que el primer término se vuelve cero para y=0 , de igual manera que el segundo término se vuelve cero para y=1. Tambien se deduce que si nuestra predicción es errónea el coste se va para el infinito (*se penaliza predicciones incorrectas con un coste elevado*).

Una de las razones del uso de esta función de coste es que es una ***función convexa*** y nos asegurará la convergencia a un mínimo global. 

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



