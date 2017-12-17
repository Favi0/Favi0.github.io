Hola, en este tutorial voy a tratar de explicaros de forma resumida, en que consiste el algoritmo de **regresión logística**.


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
> - Clearing your browser's data may **delete all your local documents!** Make sure your documents are synchronized with **Google Drive** or **Dropbox** (check out the [<i class="icon-refresh"></i> Synchronization](#synchronization) section).

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
logit(p(y=1|x))=w_0 x_1 +w_1x_1+...+w_nx_n=\sum_{i=0}^{m} w_ix_i=w^{T}x
$$

donde p(y=1|x) es la *probabilidad condicional* de que una muestra x pertenezca a la clase 1, dado su determinado vector de caracteristicas x1,..,xn. 
Notar que esta combinación lineal puede expresarse como un producto de matrices por medio de la transpuesta (dado que todas las muestras xi tienen la misma dimensión, o el mismo número de características).

> Como contamos con un conjunto de muestras X, compuestas por n características cada una, y los parámetros W del modelo..**¿no habría alguna forma de obtener una probabilidad en la salida, que es lo que realmente nos interesa?**

Esto se obtiene con la función inversa a *logit* , conocida popularmente como función *Sigmoidea* o función logística. Esta función tomará números reales en la entrada y devolverá probabilidades entre 0 y 1 en la salida.

$$
\sigma (z)=\frac{1}{1+e^{-z}}
$$
Donde z será la combinación lineal descripta previamente.

foto

Podrían utilizarse otras funciones en vez de la *sigmoidea* , pero esta presenta ciertas ventajas que nos serán útiles mas adelante:

 - Se relaciona de forma sencilla las posibilidades/chances de sucesos (**odds**) descriptas al inicio.
 - Los gradientes son fáciles de calcular.

Aprendizaje de los pesos
-------------------
