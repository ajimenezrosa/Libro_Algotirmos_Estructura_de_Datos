### Capítulo 12: Algoritmos Probabilísticos y Heurísticos

Los algoritmos probabilísticos y heurísticos desempeñan un papel fundamental en los campos de la inteligencia artificial y la optimización, ofreciendo soluciones aproximadas a problemas complejos donde los métodos exactos resultan impracticables. Este capítulo proporciona una exploración exhaustiva de los algoritmos de Monte Carlo, Las Vegas y Atlantic City, los algoritmos heurísticos y los algoritmos basados en búsqueda aleatoria. Cada sección incluye descripciones detalladas, ejemplos ilustrativos y ejercicios prácticos, facilitando una comprensión profunda y aplicable de estos métodos.

---

#### 12.1 Algoritmos de Monte Carlo, Las Vegas y Atlantic City

##### Descripción y Definición

Los algoritmos de Monte Carlo, Las Vegas y Atlantic City utilizan la aleatoriedad para resolver problemas complejos. Aunque comparten ciertas características, difieren en cómo manejan la incertidumbre y la probabilidad.

- **Algoritmos de Monte Carlo:** Utilizan la aleatoriedad para obtener resultados aproximados, especialmente útiles en simulaciones y problemas de integración. Estos algoritmos proporcionan una estimación con un nivel de precisión que aumenta con el número de muestras.

- **Algoritmos de Las Vegas:** Garantizan una solución correcta o no dan solución alguna, utilizando la aleatoriedad para buscar soluciones óptimas. La eficiencia de estos algoritmos depende de la suerte, pero siempre proporcionan una respuesta correcta si se encuentra una.

- **Algoritmos de Atlantic City:** Proporcionan soluciones correctas con alta probabilidad, ofreciendo un equilibrio entre precisión y eficiencia.

##### Ejemplos

**Ejemplo 1: Estimación del Valor de Pi con Monte Carlo**

Este ejemplo utiliza un algoritmo de Monte Carlo para estimar el valor de Pi mediante la simulación de puntos aleatorios en un cuadrado y contando cuántos caen dentro de un círculo inscrito.

```python
import random

def estimar_pi(n):
    dentro_circulo = 0
    for _ in range(n):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            dentro_circulo += 1
    return (dentro_circulo / n) * 4

print(f"Estimación de Pi: {estimar_pi(100000)}")
```

**Ejemplo 2: Algoritmo de Las Vegas para Ordenación**

Este ejemplo utiliza un algoritmo de Las Vegas para ordenar una lista de números. El algoritmo realiza permutaciones aleatorias hasta que encuentra una lista ordenada.

```python
import random

def es_ordenado(lista):
    return all(lista[i] <= lista[i + 1] for i in range(len(lista) - 1))

def ordenar_las_vegas(lista):
    while not es_ordenado(lista):
        random.shuffle(lista)
    return lista

lista = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(f"Lista ordenada: {ordenar_las_vegas(lista)}")
```

**Ejemplo 3: Algoritmo de Atlantic City para Búsqueda**

Este ejemplo utiliza un algoritmo de Atlantic City para buscar un elemento en una lista con alta probabilidad de éxito.

```python
import random

def buscar_atlantic_city(lista, elemento, n):
    for _ in range(n):
        indice = random.randint(0, len(lista) - 1)
        if lista[indice] == elemento:
            return indice
    return -1

lista = [random.randint(0, 100) for _ in range(100)]
elemento = 42
print(f"Índice del elemento {elemento}: {buscar_atlantic_city(lista, elemento, 50)}")
```

---

#### 12.2 Algoritmos Heurísticos

##### Descripción y Definición

Los algoritmos heurísticos son métodos diseñados para resolver problemas complejos de manera eficiente, utilizando estrategias de búsqueda informadas basadas en reglas empíricas o "heurísticas". Estos algoritmos no garantizan una solución óptima, pero son altamente efectivos para encontrar soluciones satisfactorias en un tiempo razonable. La principal ventaja de los algoritmos heurísticos radica en su capacidad para manejar problemas de gran escala y alta complejidad, donde los enfoques exactos resultarían impracticables debido a los elevados tiempos de computación o a la excesiva demanda de recursos..

Algunos de los enfoques heurísticos más comunes incluyen:

- **Búsqueda voraz (greedy search):** Este enfoque selecciona la mejor opción local en cada paso con la esperanza de encontrar una solución global óptima. Es simple y rápido, pero no siempre garantiza una solución óptima global debido a su naturaleza miope.
- **Búsqueda tabú (tabu search):** Utiliza una memoria a corto plazo para evitar ciclos y mejorar las soluciones a lo largo de iteraciones. Al mantener una lista de movimientos prohibidos (tabú), este enfoque puede explorar de manera más efectiva el espacio de soluciones y evitar quedar atrapado en óptimos locales.
- **Recocido simulado (simulated annealing):** Simula el proceso de recocido en metalurgia para escapar de óptimos locales y encontrar una solución global. Este algoritmo permite aceptar soluciones peores con una probabilidad que disminuye con el tiempo, facilitando la exploración de nuevas áreas del espacio de búsqueda y evitando la convergencia prematura.

##### Ejemplos

**Ejemplo 1: Búsqueda Voraz para el Problema del Cambio**

Utilizamos un algoritmo voraz para resolver el problema del cambio, buscando la mínima cantidad de monedas necesarias para alcanzar una cantidad dada.

```python
def cambio_voraz(monedas, cantidad):
    resultado = []
    monedas.sort(reverse=True)
    for moneda in monedas:
        while cantidad >= moneda:
            cantidad -= moneda
            resultado.append(moneda)
    return resultado

monedas = [25, 10, 5, 1]
cantidad = 67
print(f"Monedas necesarias: {cambio_voraz(monedas, cantidad)}")
```

**Ejemplo 2: Búsqueda Tabú para el Problema de la Mochila**

Implementamos una búsqueda tabú para resolver el problema de la mochila, maximizando el valor total de los ítems seleccionados sin exceder la capacidad de la mochila.

```python
import random

def mochila_tabú(pesos, valores, capacidad, iteraciones, tamaño_tabú):
    n = len(pesos)
    mejor_solucion = [0] * n
    mejor_valor = 0
    tabú = []
    solucion_actual = [0] * n
    valor_actual = 0

    for _ in range(iteraciones):
        vecinos = generar_vecinos(solucion_actual)
        mejor_vecino = None
        mejor_valor_vecino = 0

        for vecino in vecinos:
            if vecino not in tabú and calcular_peso(vecino, pesos) <= capacidad:
                valor_vecino = calcular_valor(vecino, valores)
                if valor_vecino > mejor_valor_vecino:
                    mejor_valor_vecino = valor_vecino
                    mejor_vecino = vecino

        if mejor_vecino is not None:
            solucion_actual = mejor_vecino
            valor_actual = mejor_valor_vecino
            tabú.append(mejor_vecino)
            if len(tabú) > tamaño_tabú:
                tabú.pop(0)
            if valor_actual > mejor_valor:
                mejor_solucion = solucion_actual
                mejor_valor = valor_actual

    return mejor_solucion, mejor_valor

def generar_vecinos(solucion):
    vecinos = []
    for i in range(len(solucion)):
        vecino = solucion[:]
        vecino[i] = 1 - vecino[i]
        vecinos.append(vecino)
    return vecinos

def calcular_peso(solucion, pesos):
    return sum([solucion[i] * pesos[i] for i in range(len(solucion))])

def calcular_valor(solucion, valores):
    return sum([solucion[i] * valores[i] for i in range(len(solucion))])

pesos = [2, 3, 4, 5]
valores = [3, 4, 5, 6]
capacidad = 5
iteraciones = 100
tamaño_tabú = 10

solucion, valor = mochila_tabú(pesos, valores, capacidad, iteraciones, tamaño_tabú)
print(f"Mejor solución: {solucion}")
print(f"Valor máximo: {valor}")
```

---

#### 12.3 Algoritmos Basados en Búsqueda Aleatoria

##### Descripción y Definición

Los algoritmos basados en búsqueda aleatoria exploran el espacio de búsqueda de manera estocástica para encontrar soluciones a problemas complejos. Estos algoritmos no siguen un camino determinista, lo que les permite escapar de óptimos locales y explorar una mayor diversidad de soluciones.

Entre los algoritmos más conocidos de esta categoría se encuentran:

- **Búsqueda aleatoria simple:** Explora el espacio de búsqueda seleccionando puntos aleatorios.
- **Búsqueda aleatoria con reinicio:** Reinicia la búsqueda aleatoria después de un número fijo de iteraciones sin mejora.
- **Algoritmos genéticos:** Utilizan mecanismos de selección, cruce y mutación para evolucionar una población de soluciones hacia el óptimo.

##### Ejemplos

**Ejemplo 1: Búsqueda Aleatoria para Optimización de Funciones**

Utilizamos una búsqueda aleatoria para optimizar una función cuadrática simple.

```python
import random

def optimizar_funcion_aleatoria(funcion, limites, iteraciones):
    mejor_solucion = None
    mejor_valor = float('inf')
    for _ in range(iteraciones):
        solucion = [random.uniform(limite[0], limite[1]) for limite in limites]
        valor = funcion(solucion)
        if valor < mejor_valor:
            mejor_valor = valor
            mejor_solucion = solucion
    return mejor_solucion, mejor_valor

def funcion(x):
    return x[0]**2 + x[1]**2

limites = [(-10, 10), (-10, 10)]
iteraciones = 1000
mejor_solucion, mejor_valor = optimizar_funcion_aleatoria(funcion, limites, iteraciones)
print(f"Mejor

 solución: {mejor_solucion}")
print(f"Mejor valor: {mejor_valor}")
```

**Ejemplo 2: Algoritmo Genético para Optimización de Funciones**

Implementamos un algoritmo genético para optimizar una función de múltiples variables.

```python
import random
from deap import base, creator, tools, algorithms

def funcion_objetivo(individual):
    return -sum((x - 5)**2 for x in individual),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", funcion_objetivo)

population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, stats=None, halloffame=None, verbose=True)
```

---

### Ejercicios

1. Implementar un algoritmo de Monte Carlo para estimar el área de un círculo inscrito en un cuadrado de lado 2.
   ```python
   import random

   def estimar_area_circulo(n):
       dentro_circulo = 0
       for _ in range(n):
           x, y = random.uniform(-1, 1), random.uniform(-1, 1)
           if x**2 + y**2 <= 1:
               dentro_circulo += 1
       return (dentro_circulo / n) * 4

   print(f"Estimación del área del círculo: {estimar_area_circulo(100000)}")
   ```

2. Implementar un algoritmo heurístico para encontrar la mínima cantidad de billetes necesarios para alcanzar una suma dada.
   ```python
   def billetes_voraz(billetes, cantidad):
       resultado = []
       billetes.sort(reverse=True)
       for billete in billetes:
           while cantidad >= billete:
               cantidad -= billete
               resultado.append(billete)
       return resultado

   billetes = [100, 50, 20, 10, 5, 1]
   cantidad = 167
   print(f"Billetes necesarios: {billetes_voraz(billetes, cantidad)}")
   ```

3. Implementar un algoritmo basado en búsqueda aleatoria para encontrar el mínimo de una función cúbica.
   ```python
   import random

   def optimizar_funcion_aleatoria(funcion, limites, iteraciones):
       mejor_solucion = None
       mejor_valor = float('inf')
       for _ in range(iteraciones):
           solucion = [random.uniform(limite[0], limite[1]) for limite in limites]
           valor = funcion(solucion)
           if valor < mejor_valor:
               mejor_valor = valor
               mejor_solucion = solucion
       return mejor_solucion, mejor_valor

   def funcion(x):
       return x[0]**3 + x[1]**3

   limites = [(-10, 10), (-10, 10)]
   iteraciones = 1000
   mejor_solucion, mejor_valor = optimizar_funcion_aleatoria(funcion, limites, iteraciones)
   print(f"Mejor solución: {mejor_solucion}")
   print(f"Mejor valor: {mejor_valor}")
   ```

4. Implementar un algoritmo de búsqueda tabú para optimizar una función cuadrática.
   ```python
   import random

   def generar_vecinos(solucion):
       vecinos = []
       for i in range(len(solucion)):
           vecino = solucion[:]
           vecino[i] = 1 - vecino[i]
           vecinos.append(vecino)
       return vecinos

   def calcular_valor(solucion):
       return sum([solucion[i] * valores[i] for i in range(len(solucion))])

   valores = [2, 3, 4, 5]
   capacidad = 5
   iteraciones = 100
   tamaño_tabú = 10

   def optimizar_tabú(valres, capacidad, iteraciones, tamaño_tabú):
       n = len(valores)
       mejor_solucion = [0] * n
       mejor_valor = 0
       tabú = []
       solucion_actual = [0] * n
       valor_actual = 0

       for _ in range(iteraciones):
           vecinos = generar_vecinos(solucion_actual)
           mejor_vecino = None
           mejor_valor_vecino = 0

           for vecino in vecinos:
               if vecino not in tabú and calcular_valor(vecino) <= capacidad:
                   valor_vecino = calcular_valor(vecino)
                   if valor_vecino > mejor_valor_vecino:
                       mejor_valor_vecino = valor_vecino
                       mejor_vecino = vecino

           if mejor_vecino es noone:
               solucion_actual = mejor_vecino
               valor_actual = mejor_valor_vecino
               tabú.append(mejor_vecino)
               if len(tabú) > tamaño_tabú:
                   tabú.pop(0)
               if valor_actual > mejor_valor:
                   mejor_solucion = solucion_actual
                   mejor_valor = valor_actual

       return mejor_solucion, mejor_valor

   solucion, valor = optimizar_tabú(valores, capacidad, iteraciones, tamaño_tabú)
   print(f"Mejor solución: {solucion}")
   print(f"Valor máximo: {valor}")
   ```

5. Implementar un algoritmo genético para maximizar la suma de los valores de un vector.
   ```python
   import random
   from deap import base, creator, tools, algorithms

   def funcion_objetivo(individual):
       return -sum((x - 5)**2 for x in individual),

   creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
   creator.create("Individual", list, fitness=creator.FitnessMin)
   toolbox = base.Toolbox()
   toolbox.register("attr_float", random.uniform, -10, 10)
   toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 5)
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   toolbox.register("mate", tools.cxBlend, alpha=0.5)
   toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
   toolbox.register("select", tools.selTournament, tournsize=3)
   toolbox.register("evaluate", funcion_objetivo)

   population = toolbox.population(n=100)
   algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, stats=None, halloffame=None, verbose=True)
   ```

---

### Examen del Capítulo

1. **¿Qué es un algoritmo de Monte Carlo?**
   - a) Un algoritmo de búsqueda
   - b) Un método probabilístico para obtener resultados aproximados
   - c) Un algoritmo de clasificación
   - d) Un tipo de estructura de datos

   - **Respuesta correcta:** b) Un método probabilístico para obtener resultados aproximados
   - **Justificación:** Los algoritmos de Monte Carlo utilizan la aleatoriedad para obtener estimaciones aproximadas de valores o comportamientos de sistemas complejos.

2. **¿Cuál es la diferencia principal entre los algoritmos de Las Vegas y los algoritmos de Atlantic City?**
   - a) Los algoritmos de Las Vegas siempre encuentran una solución óptima
   - b) Los algoritmos de Atlantic City garantizan una solución correcta con alta probabilidad
   - c) Los algoritmos de Las Vegas utilizan menos recursos
   - d) No hay diferencia entre ellos

   - **Respuesta correcta:** b) Los algoritmos de Atlantic City garantizan una solución correcta con alta probabilidad
   - **Justificación:** Los algoritmos de Las Vegas garantizan una solución correcta o no dan solución, mientras que los algoritmos de Atlantic City proporcionan soluciones correctas con alta probabilidad.

3. **¿Qué es un algoritmo heurístico?**
   - a) Un algoritmo que siempre encuentra la solución óptima
   - b) Un método basado en reglas empíricas para encontrar soluciones satisfactorias
   - c) Un algoritmo de ordenamiento
   - d) Un tipo de búsqueda binaria

   - **Respuesta correcta:** b) Un método basado en reglas empíricas para encontrar soluciones satisfactorias
   - **Justificación:** Los algoritmos heurísticos utilizan reglas empíricas o "heurísticas" para encontrar soluciones satisfactorias en problemas complejos.

4. **¿Cuál es la principal ventaja de los algoritmos de búsqueda aleatoria?**
   - a) Siempre encuentran la solución óptima
   - b) Pueden escapar de óptimos locales
   - c) Utilizan menos recursos computacionales
   - d) Son fáciles de implementar

   - **Respuesta correcta:** b) Pueden escapar de óptimos locales
   - **Justificación:** Los algoritmos de búsqueda aleatoria exploran el espacio de búsqueda de manera estocástica, lo que les permite escapar de óptimos locales y encontrar mejores soluciones.

5. **¿Qué es un

 algoritmo genético?**
   - a) Un método de optimización basado en procesos evolutivos
   - b) Un algoritmo de búsqueda binaria
   - c) Un tipo de estructura de datos
   - d) Un método de ordenamiento rápido

   - **Respuesta correcta:** a) Un método de optimización basado en procesos evolutivos
   - **Justificación:** Los algoritmos genéticos utilizan mecanismos inspirados en la evolución natural, como la selección, el cruce y la mutación, para encontrar soluciones óptimas.

6. **¿Qué es la búsqueda voraz (greedy search)?**
   - a) Un método que siempre encuentra la solución óptima
   - b) Un algoritmo que selecciona la mejor opción local en cada paso
   - c) Un tipo de búsqueda binaria
   - d) Un método de optimización global

   - **Respuesta correcta:** b) Un algoritmo que selecciona la mejor opción local en cada paso
   - **Justificación:** La búsqueda voraz selecciona la mejor opción local en cada paso con la esperanza de encontrar una solución global óptima.

7. **¿Cómo funciona el recocido simulado (simulated annealing)?**
   - a) Utiliza una estructura de datos especial para encontrar soluciones óptimas
   - b) Simula el proceso de recocido en metalurgia para escapar de óptimos locales
   - c) Siempre encuentra la solución óptima
   - d) Es un tipo de búsqueda binaria

   - **Respuesta correcta:** b) Simula el proceso de recocido en metalurgia para escapar de óptimos locales
   - **Justificación:** El recocido simulado utiliza una técnica inspirada en el proceso de recocido en metalurgia para encontrar soluciones óptimas, permitiendo movimientos hacia soluciones peores con una probabilidad decreciente.

8. **¿Cuál es la función principal de la memoria tabú en la búsqueda tabú?**
   - a) Almacenar las soluciones óptimas
   - b) Evitar ciclos y mejorar soluciones a lo largo de iteraciones
   - c) Optimizar la búsqueda binaria
   - d) Encontrar soluciones óptimas

   - **Respuesta correcta:** b) Evitar ciclos y mejorar soluciones a lo largo de iteraciones
   - **Justificación:** La memoria tabú en la búsqueda tabú almacena soluciones recientes para evitar ciclos y mejorar las soluciones a lo largo de múltiples iteraciones.

9. **¿Qué caracteriza a un algoritmo de Monte Carlo?**
   - a) Es un algoritmo de búsqueda binaria
   - b) Utiliza la aleatoriedad para obtener resultados aproximados
   - c) Siempre encuentra la solución óptima
   - d) Es un método de clasificación

   - **Respuesta correcta:** b) Utiliza la aleatoriedad para obtener resultados aproximados
   - **Justificación:** Los algoritmos de Monte Carlo se basan en la aleatoriedad para simular sistemas complejos y obtener resultados aproximados.

10. **¿Qué es la búsqueda aleatoria con reinicio?**
    - a) Un método de búsqueda binaria
    - b) Una técnica que reinicia la búsqueda aleatoria después de un número fijo de iteraciones sin mejora
    - c) Un algoritmo de ordenamiento rápido
    - d) Una estructura de datos

    - **Respuesta correcta:** b) Una técnica que reinicia la búsqueda aleatoria después de un número fijo de iteraciones sin mejora
    - **Justificación:** La búsqueda aleatoria con reinicio reinicia la búsqueda aleatoria después de un número


    ### Cierre del Capítulo

En este capítulo, hemos explorado una variedad de enfoques de algoritmos probabilísticos y heurísticos, incluyendo los algoritmos de Monte Carlo, Las Vegas y Atlantic City, así como los algoritmos heurísticos y los basados en búsqueda aleatoria. Estos métodos representan herramientas valiosas para la resolución de problemas complejos, donde los métodos deterministas tradicionales pueden no ser aplicables o efectivos.

A través de explicaciones detalladas, ejemplos prácticos y ejercicios, hemos mostrado cómo estos algoritmos pueden aplicarse en una amplia gama de áreas, desde la estimación de valores hasta la optimización de funciones y la resolución de problemas combinatorios. Los algoritmos probabilísticos y heurísticos son fundamentales en el campo de la inteligencia artificial y la optimización, proporcionando soluciones eficientes y prácticas para problemas que de otro modo serían intratables.

El conocimiento adquirido en este capítulo permite abordar problemas complejos con un enfoque innovador y flexible, utilizando técnicas que aprovechan la aleatoriedad y las reglas empíricas para encontrar soluciones satisfactorias. Al comprender y aplicar estos algoritmos, se mejora la capacidad para enfrentar desafíos en diversas disciplinas, incluyendo la ciencia de datos, la ingeniería, la economía y la investigación operativa.

### Resumen del Capítulo

En este capítulo, hemos profundizado en los algoritmos probabilísticos y heurísticos, explorando sus fundamentos, aplicaciones y ejemplos prácticos. 

- **Algoritmos de Monte Carlo, Las Vegas y Atlantic City:** Estos algoritmos utilizan la aleatoriedad para obtener resultados aproximados o probabilísticos. Los algoritmos de Monte Carlo se utilizan para la estimación y simulación, los algoritmos de Las Vegas garantizan una solución correcta o no dan solución, y los algoritmos de Atlantic City proporcionan soluciones correctas con alta probabilidad.
- **Algoritmos Heurísticos:** Los algoritmos heurísticos utilizan reglas empíricas para encontrar soluciones satisfactorias en problemas complejos. Son especialmente útiles cuando no se requiere una solución óptima, sino una que sea suficientemente buena en un tiempo razonable.
- **Algoritmos Basados en Búsqueda Aleatoria:** Estos algoritmos exploran el espacio de búsqueda de manera estocástica, permitiendo escapar de óptimos locales y mejorar las soluciones encontradas. Ejemplos incluyen la búsqueda tabú y el recocido simulado, que utilizan técnicas inspiradas en la naturaleza para optimizar soluciones.

Se presentaron varios ejemplos prácticos para ilustrar el funcionamiento de estos algoritmos, incluyendo la estimación del área de un círculo mediante el método de Monte Carlo, la optimización de funciones con algoritmos genéticos y el uso de búsqueda tabú para resolver problemas de optimización.

El capítulo también incluyó ejercicios y un examen final con preguntas de selección múltiple para consolidar los conocimientos adquiridos. Estos ejercicios permiten aplicar los conceptos y técnicas aprendidas, fortaleciendo la comprensión de los algoritmos probabilísticos y heurísticos.

En conclusión, los algoritmos probabilísticos y heurísticos son herramientas poderosas y versátiles para abordar una amplia variedad de problemas complejos. Su capacidad para encontrar soluciones eficientes y prácticas hace que sean esenciales en el arsenal de cualquier profesional en el campo de la inteligencia artificial y la optimización.