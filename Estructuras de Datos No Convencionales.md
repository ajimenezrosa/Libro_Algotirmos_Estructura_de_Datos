### Capítulo 13: Estructuras de Datos No Convencionales

Las estructuras de datos no convencionales desempeñan un papel crucial en la informática, ya que están diseñadas para abordar problemas específicos de manera altamente eficiente. Estas estructuras ofrecen soluciones alternativas y complementarias a las estructuras de datos lineales y no lineales tradicionales, brindando ventajas significativas en términos de rendimiento y optimización. En este capítulo, nos centraremos en tres estructuras de datos esenciales: Tries (Árboles de Prefijos), Tablas de Hash (Hash Tables) y Heaps (Montículos). 

Primero, exploraremos las **Tries (Árboles de Prefijos)**, que son estructuras de datos en forma de árbol utilizadas para almacenar un conjunto dinámico de cadenas, donde cada nodo representa un carácter de una cadena. Las Tries son particularmente eficientes para realizar operaciones de búsqueda, inserción y eliminación de cadenas, haciendo uso de sus propiedades jerárquicas para facilitar la navegación y manipulación de datos. Las Tries encuentran aplicaciones en sistemas de autocompletado, correctores ortográficos y otras áreas donde la búsqueda de texto rápida es crucial. En este capítulo, proporcionaremos definiciones detalladas, exploraremos sus aplicaciones prácticas y presentaremos ejemplos de implementación en Python para ilustrar cómo se pueden utilizar eficazmente.

A continuación, examinaremos las **Tablas de Hash (Hash Tables)**, que son estructuras de datos que utilizan funciones hash para mapear claves a valores, permitiendo una búsqueda, inserción y eliminación de datos extremadamente rápida. Las Tablas de Hash son fundamentales para gestionar grandes conjuntos de datos y son utilizadas en una variedad de aplicaciones que requieren acceso rápido y eficiente a los datos, como bases de datos, cachés y sistemas de almacenamiento de claves y valores. Discutiremos las técnicas para manejar colisiones, como el encadenamiento y la direccionamiento abierto, y presentaremos ejemplos prácticos en Python para demostrar cómo implementar y utilizar las Tablas de Hash.

Finalmente, exploraremos los **Heaps (Montículos)**, que son estructuras de datos especializadas que permiten la gestión eficiente de colas de prioridad. Los Heaps se utilizan ampliamente en algoritmos de ordenamiento y en la gestión de recursos en sistemas operativos, debido a su capacidad para permitir la extracción rápida del elemento mínimo o máximo. Analizaremos tanto los min-heaps como los max-heaps, explicando sus propiedades y cómo se pueden utilizar para mejorar el rendimiento de diversas operaciones. También proporcionaremos ejemplos detallados de implementación en Python para ilustrar cómo construir y manipular Heaps.

A lo largo de este capítulo, no solo ofreceremos una comprensión teórica de estas estructuras de datos, sino que también presentaremos aplicaciones prácticas y ejemplos detallados que permitirán a los lectores ver cómo estas estructuras pueden ser implementadas y utilizadas en situaciones reales. Nuestro objetivo es equipar a los lectores con el conocimiento y las habilidades necesarias para aplicar estas estructuras de datos no convencionales en el desarrollo de soluciones eficientes y robustas en Python.

---

#### 13.1 Tries (Árboles de Prefijos)

##### Descripción y Definición

Un Trie, también conocido como árbol de prefijos, es una estructura de datos en forma de árbol que se utiliza para almacenar un conjunto dinámico de cadenas, donde las claves son generalmente cadenas de caracteres. Un Trie facilita la búsqueda de palabras y prefijos, y es muy eficiente en operaciones de inserción y búsqueda.

Cada nodo en un Trie representa un carácter de la cadena. Los nodos hijos de un nodo representan posibles caracteres siguientes en las cadenas almacenadas. Un Trie completo permite verificar si una cadena es un prefijo de cualquier cadena en el Trie o si existe una cadena completa en el Trie.

##### Ejemplos

### Ejemplo 1: Implementación Básica de un Trie

#### Descripción del Código

El siguiente código implementa un Trie, una estructura de datos eficiente para almacenar y buscar cadenas de caracteres. Un Trie, también conocido como árbol de prefijos, es útil para aplicaciones como el autocompletado y los correctores ortográficos, ya que permite realizar búsquedas rápidas de palabras y prefijos.

#### Definición de las Clases

**Clase `TrieNode`:**
- **Propósito:** Representa un nodo individual en el Trie.
- **Atributos:**
  - `children`: Un diccionario que almacena los hijos del nodo actual, donde las claves son caracteres y los valores son otros `TrieNode`.
  - `is_end_of_word`: Un booleano que indica si el nodo representa el final de una palabra.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
```

**Clase `Trie`:**
- **Propósito:** Representa el Trie completo.
- **Atributos:**
  - `root`: El nodo raíz del Trie, que es una instancia de `TrieNode`.
- **Métodos:**
  - `insert(word)`: Inserta una palabra en el Trie.
  - `search(word)`: Busca una palabra en el Trie y devuelve `True` si la palabra existe, de lo contrario, devuelve `False`.
  - `starts_with(prefix)`: Comprueba si algún prefijo dado existe en el Trie.

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()
```

#### Métodos de la Clase `Trie`

**Método `insert`:**
- **Propósito:** Inserta una palabra en el Trie.
- **Descripción:** Comienza desde la raíz y para cada carácter en la palabra, comprueba si el carácter ya existe en los hijos del nodo actual. Si no existe, crea un nuevo nodo. Luego, avanza al siguiente nodo hijo. Al final de la palabra, marca el último nodo como el final de una palabra (`is_end_of_word = True`).

```python
def insert(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end_of_word = True
```

**Método `search`:**
- **Propósito:** Busca una palabra en el Trie.
- **Descripción:** Comienza desde la raíz y para cada carácter en la palabra, comprueba si el carácter existe en los hijos del nodo actual. Si en cualquier punto el carácter no existe, devuelve `False`. Si llega al final de la palabra, comprueba si el último nodo está marcado como el final de una palabra y devuelve el resultado.

```python
def search(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end_of_word
```

**Método `starts_with`:**
- **Propósito:** Comprueba si existe algún prefijo en el Trie.
- **Descripción:** Comienza desde la raíz y para cada carácter en el prefijo, comprueba si el carácter existe en los hijos del nodo actual. Si en cualquier punto el carácter no existe, devuelve `False`. Si llega al final del prefijo, devuelve `True`.

```python
def starts_with(self, prefix):
    node = self.root
    for char in prefix:
        if char not in node.children:
            return False
        node = node.children[char]
    return True
```

#### Ejemplo de Uso

En el siguiente ejemplo, se crea una instancia del Trie y se insertan las palabras "hello" y "world". Luego, se realizan varias búsquedas para comprobar la existencia de palabras completas y prefijos:

```python
# Ejemplo de uso
trie = Trie()
trie.insert("hello")
trie.insert("world")
print(trie.search("hello"))  # True
print(trie.search("world"))  # True
print(trie.search("hell"))   # False
print(trie.starts_with("wor"))  # True
print(trie.starts_with("woa"))  # False
```

- `trie.search("hello")` devuelve `True` porque "hello" se ha insertado en el Trie.
- `trie.search("world")` devuelve `True` porque "world" se ha insertado en el Trie.
- `trie.search("hell")` devuelve `False` porque aunque "hell" es un prefijo de "hello", no es una palabra completa insertada en el Trie.
- `trie.starts_with("wor")` devuelve `True` porque "wor" es un prefijo de "world".
- `trie.starts_with("woa")` devuelve `False` porque no hay ninguna palabra en el Trie que comience con "woa".


**Codigo completo*

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Ejemplo de uso
trie = Trie()
trie.insert("hello")
trie.insert("world")
print(trie.search("hello"))  # True
print(trie.search("world"))  # True
print(trie.search("hell"))   # False
print(trie.starts_with("wor"))  # True
print(trie.starts_with("woa"))  # False
```

---

#### 13.2 Tablas de Hash (Hash Tables)

##### Descripción y Definición

Una tabla de hash es una estructura de datos que asocia claves con valores. Utiliza una función hash para calcular un índice en una matriz de cubetas o slots, desde el cual se puede encontrar el valor deseado. Las tablas de hash son muy eficientes para operaciones de búsqueda, inserción y eliminación, con un tiempo de operación promedio de O(1).

El manejo de colisiones es un aspecto crucial de las tablas de hash. Las colisiones ocurren cuando dos claves distintas tienen el mismo índice. Los métodos comunes para manejar colisiones incluyen la encadenación (donde se usa una lista enlazada en cada cubeta) y la exploración lineal (donde se encuentra el siguiente cubeta disponible).

##### Ejemplos

### Ejemplo 1: Implementación Básica de una Tabla de Hash con Encadenación

Las tablas de hash son estructuras de datos que permiten una búsqueda, inserción y eliminación rápidas. Utilizan una función de hash para mapear claves a índices en una tabla, lo que permite tiempos de acceso promedio muy rápidos (O(1)). Cuando varias claves se mapean al mismo índice (colisión), se utiliza una técnica de encadenación, donde cada entrada en la tabla de hash es una lista que almacena todos los pares clave-valor que se mapean a ese índice.

#### Definición del Código

El siguiente código implementa una tabla de hash utilizando encadenación para resolver colisiones.

**Clase `HashTable`:**
- **Propósito:** Representa una tabla de hash con encadenación.
- **Atributos:**
  - `size`: El tamaño de la tabla de hash.
  - `table`: Una lista de listas, donde cada lista almacena los pares clave-valor que se mapean a ese índice.
- **Métodos:**
  - `hash_function(key)`: Calcula el índice de una clave utilizando la función de hash.
  - `insert(key, value)`: Inserta una clave y su valor en la tabla.
  - `search(key)`: Busca el valor asociado a una clave dada.
  - `delete(key)`: Elimina una clave y su valor de la tabla.

```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
```

**Constructor `__init__`:**
- **Propósito:** Inicializa la tabla de hash.
- **Parámetros:**
  - `size`: El tamaño de la tabla de hash.
- **Descripción:** Crea una lista de listas vacías con el tamaño especificado.

```python
    def hash_function(self, key):
        return hash(key) % self.size
```

**Método `hash_function`:**
- **Propósito:** Calcula el índice de una clave.
- **Parámetros:**
  - `key`: La clave para la cual se calcula el índice.
- **Descripción:** Utiliza la función de hash incorporada de Python y toma el módulo del tamaño de la tabla para obtener un índice válido.

```python
    def insert(self, key, value):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                kv[1] = value
                return
        self.table[index].append([key, value])
```

**Método `insert`:**
- **Propósito:** Inserta una clave y su valor en la tabla de hash.
- **Parámetros:**
  - `key`: La clave a insertar.
  - `value`: El valor asociado a la clave.
- **Descripción:** Calcula el índice de la clave. Si la clave ya existe en la lista en ese índice, actualiza su valor. Si no, añade un nuevo par clave-valor a la lista.

```python
    def search(self, key):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                return kv[1]
        return None
```

**Método `search`:**
- **Propósito:** Busca el valor asociado a una clave dada.
- **Parámetros:**
  - `key`: La clave a buscar.
- **Descripción:** Calcula el índice de la clave y recorre la lista en ese índice para encontrar la clave. Si la encuentra, devuelve su valor. Si no, devuelve `None`.

```python
    def delete(self, key):
        index = self.hash_function(key)
        for i, kv in enumerate(self.table[index]):
            if kv[0] == key:
                del self.table[index][i]
                return
```

**Método `delete`:**
- **Propósito:** Elimina una clave y su valor de la tabla de hash.
- **Parámetros:**
  - `key`: La clave a eliminar.
- **Descripción:** Calcula el índice de la clave y recorre la lista en ese índice para encontrar la clave. Si la encuentra, elimina el par clave-valor de la lista.

#### Ejemplo de Uso

En el siguiente ejemplo, se crea una instancia de la tabla de hash y se realizan operaciones de inserción, búsqueda y eliminación para demostrar su funcionalidad.

```python
# Ejemplo de uso
hash_table = HashTable(10)
hash_table.insert("name", "Alice")
hash_table.insert("age", 30)
print(hash_table.search("name"))  # Output: Alice
print(hash_table.search("age"))   # Output: 30
hash_table.delete("age")
print(hash_table.search("age"))   # Output: None
```

- **`hash_table.insert("name", "Alice")`:** Inserta la clave `"name"` con el valor `"Alice"` en la tabla de hash.
- **`hash_table.insert("age", 30)`:** Inserta la clave `"age"` con el valor `30` en la tabla de hash.
- **`hash_table.search("name")`:** Devuelve `"Alice"` porque la clave `"name"` está presente en la tabla de hash.
- **`hash_table.search("age")`:** Devuelve `30` porque la clave `"age"` está presente en la tabla de hash.
- **`hash_table.delete("age")`:** Elimina la clave `"age"` de la tabla de hash.
- **`hash_table.search("age")`:** Devuelve `None` porque la clave `"age"` ha sido eliminada de la tabla de hash.

Este ejemplo muestra cómo se pueden usar tablas de hash con encadenación para manejar colisiones y realizar operaciones eficientes de inserción, búsqueda y eliminación.

**Codigo Completo**

```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                kv[1] = value
                return
        self.table[index].append([key, value])

    def search(self, key):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                return kv[1]
        return None

    def delete(self, key):
        index = self.hash_function(key)
        for i, kv in enumerate(self.table[index]):
            if kv[0] == key:
                del self.table[index][i]
                return

# Ejemplo de uso
hash_table = HashTable(10)
hash_table.insert("name", "Alice")
hash_table.insert("age", 30)
print(hash_table.search("name"))  # Alice
print(hash_table.search("age"))   # 30
hash_table.delete("age")
print(hash_table.search("age"))   # None
```

---

#### 13.3 Heap (Montículos)

##### Descripción y Definición

Un heap es una estructura de datos en forma de árbol binario que satisface la propiedad del heap. En un max-heap, el valor de cada nodo es mayor o igual al valor de sus hijos, y en un min-heap, el valor de cada nodo es menor o igual al valor de sus hijos. Los heaps son comúnmente utilizados para implementar colas de prioridad.

La inserción y eliminación en un heap tienen una complejidad de O(log n), lo que los hace muy eficientes para aplicaciones que requieren acceso rápido a los elementos mínimos o máximos.

##### Ejemplos

### Ejemplo 1: Implementación de un Min-Heap

Un Min-Heap es una estructura de datos en la que cada nodo es menor o igual que sus hijos. Esto asegura que el elemento más pequeño siempre se encuentre en la raíz del heap. Los Min-Heaps son útiles para implementar colas de prioridad, algoritmos de gráficos como Dijkstra y muchas otras aplicaciones.

#### Definición del Código

El siguiente código implementa un Min-Heap utilizando la biblioteca `heapq` de Python, que proporciona una manera eficiente de manejar las operaciones de heap.

**Clase `MinHeap`:**
- **Propósito:** Representa un Min-Heap.
- **Atributos:**
  - `heap`: Una lista que almacena los elementos del heap.
- **Métodos:**
  - `__init__`: Inicializa una instancia vacía del Min-Heap.
  - `insert(val)`: Inserta un nuevo valor en el heap.
  - `extract_min()`: Extrae y devuelve el elemento mínimo del heap.
  - `get_min()`: Devuelve el elemento mínimo del heap sin extraerlo.

```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []
```

**Constructor `__init__`:**
- **Propósito:** Inicializa una lista vacía para almacenar los elementos del heap.
- **Descripción:** Cuando se crea una instancia de `MinHeap`, se inicializa una lista vacía `heap`.

```python
    def insert(self, val):
        heapq.heappush(self.heap, val)
```

**Método `insert`:**
- **Propósito:** Inserta un nuevo valor en el heap.
- **Parámetros:**
  - `val`: El valor a insertar en el heap.
- **Descripción:** Utiliza la función `heappush` de la biblioteca `heapq` para añadir el nuevo valor al heap, manteniendo la propiedad del Min-Heap.

```python
    def extract_min(self):
        return heapq.heappop(self.heap)
```

**Método `extract_min`:**
- **Propósito:** Extrae y devuelve el elemento mínimo del heap.
- **Descripción:** Utiliza la función `heappop` de la biblioteca `heapq` para eliminar y devolver el elemento mínimo del heap. Esto garantiza que el heap se reestructure para mantener la propiedad del Min-Heap.

```python
    def get_min(self):
        return self.heap[0]
```

**Método `get_min`:**
- **Propósito:** Devuelve el elemento mínimo del heap sin extraerlo.
- **Descripción:** Accede al primer elemento de la lista `heap`, que siempre es el elemento mínimo en un Min-Heap.

#### Ejemplo de Uso

En el siguiente ejemplo, se crea una instancia de `MinHeap` y se realizan operaciones de inserción, extracción y obtención del valor mínimo para demostrar su funcionalidad.

```python
# Ejemplo de uso
min_heap = MinHeap()
min_heap.insert(3)
min_heap.insert(1)
min_heap.insert(6)
print(min_heap.get_min())  # Output: 1
print(min_heap.extract_min())  # Output: 1
print(min_heap.get_min())  # Output: 3
```

- **`min_heap.insert(3)`:** Inserta el valor `3` en el heap.
- **`min_heap.insert(1)`:** Inserta el valor `1` en el heap.
- **`min_heap.insert(6)`:** Inserta el valor `6` en el heap.
- **`min_heap.get_min()`:** Devuelve el valor mínimo del heap, que es `1`.
- **`min_heap.extract_min()`:** Extrae y devuelve el valor mínimo del heap, que es `1`, y reestructura el heap para mantener la propiedad del Min-Heap.
- **`min_heap.get_min()`:** Devuelve el nuevo valor mínimo del heap, que es `3`.

Este ejemplo muestra cómo se pueden usar los Min-Heaps para manejar operaciones de inserción, extracción y obtención de valores mínimos de manera eficiente.

**Codigo Completo**

```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        heapq.heappush(self.heap, val)

    def extract_min(self):
        return heapq.heappop(self.heap)

    def get_min(self):
        return self.heap[0]

# Ejemplo de uso
min_heap = MinHeap()
min_heap.insert(3)
min_heap.insert(1)
min_heap.insert(6)
print(min_heap.get_min())  # 1
print(min_heap.extract_min())  # 1
print(min_heap.get_min())  # 3
```

---

### Ejercicios

1.  Implementar una función para verificar si una palabra está en un Trie

El siguiente código define una función que permite verificar si una palabra está presente en un Trie. Un Trie es una estructura de datos de árbol utilizada principalmente para almacenar un conjunto dinámico de cadenas, donde las claves son cadenas de caracteres.

#### Descripción del Código

**Función `buscar_en_trie`:**
- **Propósito:** Verifica si una palabra está presente en un Trie.
- **Parámetros:**
  - `trie`: Una instancia de la clase `Trie`.
  - `palabra`: La palabra que se desea buscar en el Trie.
- **Descripción:** La función utiliza el método `search` del Trie para verificar si la palabra está presente y devuelve `True` si la palabra se encuentra, y `False` en caso contrario.

```python
def buscar_en_trie(trie, palabra):
    return trie.search(palabra)

# Ejemplo de uso
trie = Trie()
trie.insert("algoritmo")
trie.insert("estructura")
print(buscar_en_trie(trie, "algoritmo"))  # True
print(buscar_en_trie(trie, "algoritmos"))  # False
```

#### Ejemplo de Uso

En el siguiente ejemplo, se crea una instancia de `Trie`, se insertan dos palabras ("algoritmo" y "estructura") y se utilizan las funciones `insert` y `buscar_en_trie` para verificar la presencia de las palabras en el Trie.

1. **`trie.insert("algoritmo")`:** Inserta la palabra "algoritmo" en el Trie.
2. **`trie.insert("estructura")`:** Inserta la palabra "estructura" en el Trie.
3. **`buscar_en_trie(trie, "algoritmo")`:** Verifica si la palabra "algoritmo" está presente en el Trie. Devuelve `True` porque la palabra fue insertada.
4. **`buscar_en_trie(trie, "algoritmos")`:** Verifica si la palabra "algoritmos" está presente en el Trie. Devuelve `False` porque la palabra no fue insertada (nótese que "algoritmos" es diferente de "algoritmo" por la 's' adicional).

Este ejemplo demuestra cómo se pueden insertar palabras en un Trie y cómo verificar su existencia utilizando una función de búsqueda. El Trie es útil para tareas como la autocompletación y la corrección ortográfica, donde se necesita una búsqueda rápida y eficiente de palabras en un gran conjunto de cadenas.

2. Descripción General del Código: Encontrar Todas las Palabras en un Trie que Comiencen con un Prefijo Dado

El siguiente código define una función que permite encontrar todas las palabras almacenadas en un Trie que comienzan con un prefijo específico. Un Trie es una estructura de datos de árbol utilizada para almacenar y buscar cadenas de caracteres de manera eficiente.

#### Descripción del Código

**Función `buscar_por_prefijo`:**
- **Propósito:** Encuentra todas las palabras en un Trie que comienzan con un prefijo dado.
- **Parámetros:**
  - `trie`: Una instancia de la clase `Trie`.
  - `prefijo`: El prefijo con el que deben comenzar las palabras buscadas.
- **Descripción:** La función primero navega por el Trie hasta encontrar el nodo que corresponde al último carácter del prefijo. Luego, realiza una búsqueda en profundidad (DFS) a partir de ese nodo para encontrar todas las palabras completas que comienzan con el prefijo.

```python
def buscar_por_prefijo(trie, prefijo):
    def dfs(node, prefijo):
        palabras = []
        if node.is_end_of_word:
            palabras.append(prefijo)
        for char, child_node in node.children.items():
            palabras.extend(dfs(child_node, prefijo + char))
        return palabras

    node = trie.root
    for char in prefijo:
        if char not in node.children:
            return []
        node = node.children[char]
    return dfs(node, prefijo)

# Ejemplo de uso
trie = Trie()
trie.insert("algoritmo")
trie.insert("algoritmos")
trie.insert("algoritmica")
trie.insert("estructura")
trie.insert("estrategia")
print(buscar_por_prefijo(trie, "algo"))  # ['algoritmo', 'algoritmos', 'algoritmica']
print(buscar_por_prefijo(trie, "estru"))  # ['estructura']
print(buscar_por_prefijo(trie, "estrat"))  # ['estrategia']
```

#### Ejemplo de Uso

En el ejemplo, se crea una instancia de `Trie`, se insertan varias palabras en el Trie, y se utiliza la función `buscar_por_prefijo` para encontrar todas las palabras que comienzan con los prefijos "algo", "estru" y "estrat".

1. **`trie.insert("algoritmo")`, `trie.insert("algoritmos")`, `trie.insert("algoritmica")`:** Inserta las palabras "algoritmo", "algoritmos" y "algoritmica" en el Trie.
2. **`trie.insert("estructura")`, `trie.insert("estrategia")`:** Inserta las palabras "estructura" y "estrategia" en el Trie.
3. **`buscar_por_prefijo(trie, "algo")`:** Encuentra todas las palabras que comienzan con el prefijo "algo". Devuelve `['algoritmo', 'algoritmos', 'algoritmica']`.
4. **`buscar_por_prefijo(trie, "estru")`:** Encuentra todas las palabras que comienzan con el prefijo "estru". Devuelve `['estructura']`.
5. **`buscar_por_prefijo(trie, "estrat")`:** Encuentra todas las palabras que comienzan con el prefijo "estrat". Devuelve `['estrategia']`.

Este ejemplo muestra cómo se pueden utilizar Tries para realizar búsquedas eficientes de palabras basadas en prefijos, una funcionalidad útil en aplicaciones como la autocompletación de texto y los sistemas de búsqueda.


3. Manejar Colisiones en una Tabla de Hash Usando Exploración Lineal

El siguiente código define una función para manejar colisiones en una tabla de hash utilizando el método de exploración lineal. Las tablas de hash son estructuras de datos que permiten la inserción, búsqueda y eliminación de elementos de manera eficiente. Sin embargo, cuando dos claves producen el mismo índice en la tabla (una colisión), se deben emplear técnicas para resolver esta situación. La exploración lineal es una de esas técnicas.

#### Descripción del Código

**Clase `HashTable`:**
- **Propósito:** Implementa una tabla de hash con manejo de colisiones mediante exploración lineal.
- **Atributos:**
  - `size`: Tamaño de la tabla de hash.
  - `table`: Lista que almacena los elementos de la tabla de hash.
- **Métodos:**
  - **`hash_function(key)`:** Calcula el índice de la tabla de hash para una clave dada.
  - **`insert(key, value)`:** Inserta una clave y su valor asociado en la tabla de hash. Si ocurre una colisión, busca la siguiente posición libre utilizando exploración lineal.
  - **`search(key)`:** Busca una clave en la tabla de hash y devuelve su valor asociado. Utiliza exploración lineal para manejar colisiones.
  - **`delete(key)`:** Elimina una clave y su valor asociado de la tabla de hash. Utiliza exploración lineal para encontrar la clave.

```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        original_index = index
        while self.table[index] is not None:
            if self.table[index][0] == key:
                self.table[index][1] = value
                return
            index = (index + 1) % self.size
            if index == original_index:
                raise Exception("Hash table is full")
        self.table[index] = [key, value]

    def search(self, key):
        index = self.hash_function(key)
        original_index = index
        while self.table[index] is not None:
            if self.table[index][0] == key:
                return self.table[index][1]
            index = (index + 1) % self.size
            if index == original_index:
                return None
        return None

    def delete(self, key):
        index = self.hash_function(key)
        original_index = index
        while self.table[index] is not None:
            if self.table[index][0] == key:
                self.table[index] = None
                return
            index = (index + 1) % self.size
            if index == original_index:
                return

# Ejemplo de uso
hash_table = HashTable(10)
hash_table.insert("name", "Alice")
hash_table.insert("age", 30)
print(hash_table.search("name"))  # Alice
print(hash_table.search("age"))   # 30
hash_table.delete("age")
print(hash_table.search("age"))   # None
```

#### Explicación General

El código implementa una tabla de hash con manejo de colisiones usando exploración lineal. Aquí se describen los componentes clave:

1. **`__init__(self, size)`:** Inicializa una tabla de hash con un tamaño especificado. La tabla es una lista de tamaño `size`, inicialmente llena de `None`.

2. **`hash_function(self, key)`:** Calcula el índice de la tabla de hash para una clave dada usando la función `hash` de Python y el operador módulo (`%`).

3. **`insert(self, key, value)`:** Inserta una clave y su valor asociado en la tabla de hash. Si el índice calculado ya está ocupado, la función usa exploración lineal para encontrar la siguiente posición libre.

4. **`search(self, key)`:** Busca una clave en la tabla de hash y devuelve su valor asociado. Si la clave no está en el índice calculado, la función utiliza exploración lineal para buscar en las siguientes posiciones.

5. **`delete(self, key)`:** Elimina una clave y su valor asociado de la tabla de hash. Si la clave no está en el índice calculado, la función usa exploración lineal para buscar y eliminar la clave en las siguientes posiciones.

#### Ejemplo de Uso

En el ejemplo de uso, se demuestra cómo insertar, buscar y eliminar elementos en la tabla de hash:

1. **`hash_table.insert("name", "Alice")`:** Inserta la clave "name" con el valor "Alice".
2. **`hash_table.insert("age", 30)`:** Inserta la clave "age" con el valor `30`.
3. **`hash_table.search("name")`:** Busca la clave "name" y devuelve el valor asociado "Alice".
4. **`hash_table.search("age")`:** Busca la clave "age" y devuelve el valor asociado `30`.
5. **`hash_table.delete("age")`:** Elimina la clave "age" y su valor asociado.
6. **`hash_table.search("age")`:** Busca la clave "age" y devuelve `None` porque la clave ha sido eliminada.

Este ejemplo muestra cómo se pueden manejar colisiones en una tabla de hash utilizando exploración lineal para asegurar que cada clave se inserte, busque y elimine correctamente, incluso cuando ocurren colisiones.


4. **Implementar una función para verificar la propiedad de un heap (montículo).**
##### Descripción General del Código: Verificar la Propiedad de un Heap (Montículo)

El siguiente código define una función para verificar si un array (arreglo) dado cumple con la propiedad de un heap (montículo). Los heaps son estructuras de datos especializadas que se utilizan comúnmente para implementar colas de prioridad y en algoritmos de ordenamiento, como el heapsort. Un min-heap es una estructura en la cual cada elemento es menor o igual que sus hijos, mientras que un max-heap es lo opuesto.

#### Descripción del Código

**Función `es_heap(array)`:**
- **Propósito:** Verificar si un array cumple con la propiedad de un min-heap.
- **Parámetros:**
  - `array`: Lista de elementos que se desea verificar.
- **Retorno:** 
  - `True` si el array cumple con la propiedad de un min-heap.
  - `False` si no cumple con la propiedad de un min-heap.
- **Proceso:**
  - Itera a través de cada elemento del array.
  - Verifica si el elemento actual es mayor que cualquiera de sus hijos. Si encuentra un elemento mayor que alguno de sus hijos, devuelve `False`.
  - Si no se encuentra ninguna violación de la propiedad de heap, devuelve `True`.

#### Ejemplo de Uso

```python
def es_heap(array):
    for i in range(len(array)):
        if 2 * i + 1 < len(array) and array[i] > array[2 * i + 1]:
            return False
        if 2 * i + 2 < len(array) and array[i] > array[2 * i + 2]:
            return False
    return True

# Ejemplo de uso
print(es_heap([1, 3, 6, 5, 9, 8]))  # True
print(es_heap([10, 9, 8, 7, 6, 5]))  # False
```

#### Explicación del Ejemplo

1. **`es_heap([1, 3, 6, 5, 9, 8])`:**
   - Este array representa un min-heap. 
   - Para cada elemento, se verifica que no sea mayor que sus hijos. Por ejemplo, el primer elemento `1` no es mayor que sus hijos `3` y `6`, el segundo elemento `3` no es mayor que sus hijos `5` y `9`, y así sucesivamente.
   - Dado que todas las comparaciones cumplen con la propiedad de min-heap, la función devuelve `True`.

2. **`es_heap([10, 9, 8, 7, 6, 5])`:**
   - Este array no representa un min-heap.
   - Al verificar el primer elemento `10`, se encuentra que es mayor que sus hijos `9` y `8`.
   - Debido a esta violación de la propiedad de min-heap, la función devuelve `False`.

#### Conclusión

Este código proporciona una forma eficiente de verificar si un array cumple con la propiedad de un min-heap. Esta verificación es útil en diversas aplicaciones, como la implementación de colas de prioridad, el ordenamiento con heapsort y la manipulación de estructuras de datos que requieren la propiedad de heap para funcionar correctamente.

5. **Implementar un Max-Heap y añadir una función para extraer el elemento máximo.**

##### Descripción General del Código: Implementar un Max-Heap y Extraer el Elemento Máximo

El siguiente código define una clase para implementar un Max-Heap utilizando las funcionalidades de heap proporcionadas por la biblioteca `heapq` de Python. Un Max-Heap es una estructura de datos especializada en la que el valor máximo siempre se encuentra en la raíz del heap. Este tipo de estructura es útil para gestionar colas de prioridad, donde se requiere acceso rápido al elemento de mayor prioridad.

#### Descripción del Código

**Clase `MaxHeap`:**
- **Propósito:** Implementar un Max-Heap, que permite insertar elementos, extraer el máximo y obtener el valor máximo.
- **Atributos:**
  - `heap`: Lista que almacena los elementos del heap. Se usa la convención de almacenar los valores negados para aprovechar las funciones de Min-Heap de `heapq`.
- **Métodos:**
  - `__init__`: Inicializa un heap vacío.
  - `insert(val)`: Inserta un nuevo valor en el heap. El valor se inserta negado para mantener la propiedad del Max-Heap utilizando `heapq.heappush`.
  - `extract_max()`: Extrae y devuelve el valor máximo del heap. Se utiliza `heapq.heappop` para extraer el valor mínimo (negado) y luego se devuelve el valor positivo.
  - `get_max()`: Devuelve el valor máximo del heap sin extraerlo. Accede al primer elemento de la lista (negado) y lo devuelve en su forma positiva.

#### Ejemplo de Uso

```python
import heapq

class MaxHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        heapq.heappush(self.heap, -val)

    def extract_max(self):
        return -heapq.heappop(self.heap)

    def get_max(self):
        return -self.heap[0]

# Ejemplo de uso
max_heap = MaxHeap()
max_heap.insert(3)
max_heap.insert(1)
max_heap.insert(6)
print(max_heap.get_max())  # 6
print(max_heap.extract_max())  # 6
print(max_heap.get_max())  # 3
```

#### Explicación del Ejemplo

1. **Inicialización del Max-Heap:**
   - `max_heap = MaxHeap()`: Se crea una instancia de `MaxHeap`, inicializando un heap vacío.

2. **Inserción de Elementos:**
   - `max_heap.insert(3)`: Se inserta el valor `3` en el heap. Internamente, se almacena como `-3`.
   - `max_heap.insert(1)`: Se inserta el valor `1` en el heap. Internamente, se almacena como `-1`.
   - `max_heap.insert(6)`: Se inserta el valor `6` en el heap. Internamente, se almacena como `-6`.

3. **Obtención del Elemento Máximo:**
   - `print(max_heap.get_max())`: Devuelve `6`, que es el valor máximo en el heap. El valor `-6` se convierte a `6` antes de ser devuelto.

4. **Extracción del Elemento Máximo:**
   - `print(max_heap.extract_max())`: Extrae y devuelve `6`, que es el valor máximo en el heap. Internamente, se elimina `-6` de la lista.
   - `print(max_heap.get_max())`: Después de extraer `6`, el siguiente valor máximo es `3`. Se devuelve `3`.

#### Conclusión

Este código proporciona una implementación eficiente de un Max-Heap utilizando la biblioteca `heapq` de Python. Permite insertar elementos, obtener el valor máximo y extraer el valor máximo con operaciones de tiempo logarítmico. La utilización de valores negados permite mantener la propiedad del Max-Heap aprovechando las funciones nativas de Min-Heap de `heapq`. Esta implementación es útil en aplicaciones que requieren gestión de colas de prioridad y acceso rápido al elemento de mayor valor.


---

### Examen del Capítulo

1. **¿Qué es un Trie?**
   - a) Una estructura de datos lineal
   - b) Un árbol binario de búsqueda
   - c) Un árbol de prefijos utilizado para almacenar cadenas
   - d) Una lista enlazada

   **Respuesta correcta:** c) Un árbol de prefijos utilizado para almacenar cadenas
   **Justificación:** Un Trie es una estructura de datos que almacena cadenas, donde cada nodo representa un carácter y los nodos hijos representan posibles caracteres siguientes.

2. **¿Cuál es la función principal de una tabla de hash?**
   - a) Ordenar datos
   - b) Almacenar datos en una estructura de árbol
   - c) Asociar claves con valores usando una función hash
   - d) Realizar búsquedas binarias

   **Respuesta correcta:** c) Asociar claves con valores usando una función hash
   **Justificación:** Una tabla de hash asocia claves con valores utilizando una función hash para calcular el índice de almacenamiento.

3. **¿Qué es un heap?**
   - a) Una estructura de datos lineal
   - b) Un árbol binario que satisface la propiedad del heap
   - c) Una lista enlazada doblemente
   - d) Una tabla de dispersión

   **Respuesta correcta:** b) Un árbol binario que satisface la propiedad del heap
   **Justificación:** Un heap es una estructura de datos en forma de árbol binario que satisface la propiedad del heap, donde cada nodo es mayor o menor que sus hijos dependiendo del tipo de heap.

4. **¿Cuál es la complejidad de tiempo promedio de las operaciones en una tabla de hash?**
   - a) O(n)
   - b) O(log n)
   - c) O(1)
   - d) O(n log n)

   **Respuesta correcta:** c) O(1)
   **Justificación:** Las operaciones de búsqueda, inserción y eliminación en una tabla de hash tienen un tiempo promedio de O(1) debido a la dispersión uniforme de claves.

5. **¿Cómo maneja las colisiones una tabla de hash con encadenación?**
   - a) Usa listas enlazadas en cada cubeta para almacenar múltiples claves
   - b) Encuentra la siguiente cubeta disponible
   - c) Recalcula el índice
   - d) No maneja colisiones

   **Respuesta correcta:** a) Usa listas enlazadas en cada cubeta para almacenar múltiples claves
   **Justificación:** La encadenación maneja colisiones usando listas enlazadas en cada cubeta para almacenar múltiples claves que tienen el mismo índice.

6. **¿Qué estructura de datos se utiliza comúnmente para implementar una cola de prioridad?**
   - a) Lista enlazada
   - b) Árbol binario de búsqueda
   - c) Heap (montículo)
   - d) Tabla de hash

   **Respuesta correcta:** c) Heap (montículo)
   **Justificación:** Los heaps se utilizan comúnmente para implementar colas de prioridad debido a su eficiencia en la inserción y extracción de elementos con mayor o menor prioridad.

7. **¿Qué garantiza un algoritmo de Las Vegas?**
   - a) Siempre proporciona una solución correcta
   - b) Proporciona una solución correcta con alta probabilidad
   - c) Proporciona una solución óptima o no da solución alguna
   - d) Proporciona una solución aproximada

   **Respuesta correcta:** c) Proporciona una solución óptima o no da solución alguna
   **Justificación:** Los algoritmos de Las Vegas garantizan una solución correcta o no dan solución alguna, utilizando la aleatoriedad para buscar soluciones óptimas.

8. **¿Qué propiedad deben cumplir los nodos en un max-heap?**
   - a) El valor de cada nodo es menor o igual al valor de sus hijos
   - b) El valor de cada nodo es mayor o igual al valor de sus hijos
   - c) Todos los nodos tienen dos hijos
   - d) El valor de cada nodo es distinto del valor de sus hijos

   **Respuesta correcta:** b) El valor de cada nodo es mayor o igual al valor de sus hijos
   **Justificación:** En un max-heap, el valor de cada nodo es mayor o igual al valor de sus hijos, lo que garantiza que el valor máximo se encuentra en la raíz.

9. **¿Cuál es la principal ventaja de utilizar un Trie?**
   - a) Espacio de almacenamiento reducido
   - b) Eficiencia en la búsqueda de cadenas y prefijos
   - c) Facilita la ordenación de datos
   - d) Simplifica la implementación de árboles binarios

   **Respuesta correcta:** b) Eficiencia en la búsqueda de cadenas y prefijos
   **Justificación:** Los Tries son muy eficientes para la búsqueda de palabras y prefijos, permitiendo operaciones rápidas de inserción y búsqueda.

10. **¿Cuál es la complejidad de tiempo para la inserción en un heap?**
    - a) O(1)
    - b) O(n)
    - c) O(log n)
    - d) O(n log n)

    **Respuesta correcta:** c) O(log n)
    **Justificación:** La inserción en un heap tiene una complejidad de O(log n), ya que puede requerir la reorganización del heap para mantener la propiedad del heap.

11. **¿Cómo maneja las colisiones una tabla de hash con exploración lineal?**
    - a) Usa listas enlazadas en cada cubeta
    - b) Encuentra la siguiente cubeta disponible
    - c) Recalcula el índice
    - d) No maneja colisiones

    **Respuesta correcta:** b) Encuentra la siguiente cubeta disponible
    **Justificación:** La exploración lineal maneja colisiones encontrando la siguiente cubeta disponible en la tabla de hash.

12. **¿Cuál es la principal aplicación de un heap?**
    - a) Almacenamiento de datos ordenados
    - b) Implementación de colas de prioridad
    - c) Optimización de búsquedas
    - d) Manejo de colisiones en tablas de hash

    **Respuesta correcta:** b) Implementación de colas de prioridad
    **Justificación:** Los heaps se utilizan principalmente para implementar colas de prioridad debido a su eficiencia en la inserción y extracción de elementos con prioridad.

13. **¿Qué garantiza un algoritmo de Monte Carlo?**
    - a) Proporciona una solución correcta con alta probabilidad
    - b) Siempre proporciona una solución correcta
    - c) Proporciona una solución óptima o no da solución alguna
    - d) Proporciona una solución aproximada

    **Respuesta correcta:** d) Proporciona una solución aproximada
    **Justificación:** Los algoritmos de Monte Carlo proporcionan soluciones aproximadas utilizando la aleatoriedad para obtener resultados con un nivel de precisión que aumenta con el número de muestras.

14. **¿Cuál es la principal ventaja de una tabla de hash?**
    - a) Fácil de implementar
    - b) Búsqueda, inserción y eliminación rápidas
    - c) Eficiencia en la ordenación de datos
    - d) Reducción del espacio de almacenamiento

    **Respuesta correcta:** b) Búsqueda, inserción y eliminación rápidas
    **Justificación:** Las tablas de hash son muy eficientes para operaciones de búsqueda, inserción y eliminación, con un tiempo promedio de O(1).

15. **¿Qué propiedad deben cumplir los nodos en un min-heap?**
    - a) El valor de cada nodo es menor o igual al valor de sus hijos
    - b) El valor de cada nodo es mayor o igual al valor de sus hijos
    - c) Todos los nodos tienen dos hijos
    - d) El valor de cada nodo es distinto del valor de sus hijos

    **Respuesta correcta:** a) El valor de cada nodo es menor o igual al valor de sus hijos
    **Justificación:** En un min-heap, el valor de cada nodo es menor o igual al valor de sus hijos, lo que garantiza que el valor mínimo se encuentra en la raíz.

---



### Cierre del Capítulo

En este capítulo, hemos explorado las estructuras de datos no convencionales, específicamente Tries, Tablas de Hash y Heaps. Estas estructuras de datos ofrecen soluciones eficientes y flexibles para una variedad de problemas computacionales, desde la búsqueda rápida de cadenas y la gestión de claves-valor hasta la implementación de colas de prioridad. La comprensión y aplicación de estas estructuras son fundamentales para cualquier programador que busque optimizar el rendimiento y la eficiencia de sus algoritmos. A través de ejemplos prácticos y ejercicios, hemos visto cómo implementar y utilizar estas estructuras en Python, proporcionando una base sólida para abordar problemas más complejos en el futuro.

### Resumen del Capítulo

En este capítulo, hemos explorado detalladamente las estructuras de datos no convencionales, enfocándonos en Tries, Tablas de Hash y Heaps. Estas estructuras de datos son esenciales para resolver una amplia gama de problemas computacionales de manera eficiente y flexible. Cada una de estas estructuras ofrece ventajas únicas y se adapta a diferentes tipos de aplicaciones, desde la búsqueda rápida de cadenas y la gestión de pares clave-valor hasta la implementación de colas de prioridad.

#### Tries (Árboles de Prefijos)

Los Tries son especialmente útiles para aplicaciones que requieren búsquedas rápidas y eficientes de cadenas, como los motores de búsqueda y los sistemas de autocompletado. Su estructura permite insertar, buscar y verificar prefijos de palabras de manera extremadamente rápida, lo que los convierte en una herramienta indispensable para optimizar el rendimiento en estas tareas.

#### Tablas de Hash (Hash Tables)

Las Tablas de Hash son cruciales para la gestión de datos que necesitan acceso rápido y eficiente. Utilizan funciones hash para mapear claves a ubicaciones específicas en una tabla, permitiendo operaciones de búsqueda, inserción y eliminación en tiempo constante. Esto las hace ideales para aplicaciones como bases de datos, cachés y cualquier sistema que requiera una gestión eficiente de pares clave-valor.

#### Heaps (Montículos)

Los Heaps son fundamentales para la implementación de colas de prioridad, donde los elementos con la mayor o menor prioridad necesitan ser procesados primero. Ya sea un Min-Heap o un Max-Heap, estas estructuras permiten insertar elementos y extraer el elemento de mayor o menor prioridad de manera eficiente, siendo útiles en algoritmos de planificación de tareas y gestión de eventos en sistemas en tiempo real.

### Importancia de las Estructuras de Datos No Convencionales

La comprensión y aplicación de estas estructuras de datos no convencionales son fundamentales para cualquier programador que busque optimizar el rendimiento y la eficiencia de sus algoritmos. Cada estructura ofrece soluciones específicas y eficientes para problemas que serían complejos de manejar con estructuras de datos tradicionales. A través de ejemplos prácticos y ejercicios, hemos demostrado cómo implementar y utilizar estas estructuras en Python, proporcionando una base sólida para abordar problemas más complejos en el futuro.

### Hacia el Futuro

Al dominar Tries, Tablas de Hash y Heaps, los programadores estarán equipados con herramientas poderosas para resolver una amplia variedad de desafíos computacionales. Estas estructuras de datos no solo mejoran el rendimiento de los algoritmos, sino que también abren la puerta a nuevas formas de abordar y solucionar problemas en diversas aplicaciones. Con esta base, los lectores están preparados para explorar y aplicar técnicas avanzadas en el mundo de la programación y el diseño de algoritmos.

# 


