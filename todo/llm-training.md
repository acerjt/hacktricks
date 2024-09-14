# LLM Training

## Tokenizing

Tokenizing consiste en separar los datos en fragmentos específicos y asignarles IDs (números) específicos.\
Un tokenizador muy simple para textos podría ser simplemente obtener cada palabra de un texto por separado, así como los símbolos de puntuación y eliminar los espacios.\
Por lo tanto, `"Hello, world!"` sería: `["Hello", ",", "world", "!"]`

Luego, para asignar a cada una de las palabras y símbolos un ID de token (número), es necesario crear el **vocabulario** del tokenizador. Si estás tokenizando, por ejemplo, un libro, esto podría ser **todas las diferentes palabras del libro** en orden alfabético con algunos tokens adicionales como:

* `[BOS] (Inicio de secuencia)`: Colocado al principio de un texto, indica el inicio de un texto (se utiliza para separar textos no relacionados).
* `[EOS] (Fin de secuencia)`: Colocado al final de un texto, indica el final de un texto (se utiliza para separar textos no relacionados).
* `[PAD] (relleno)`: Cuando el tamaño de un lote es mayor que uno (usualmente), este token se utiliza para aumentar la longitud de ese lote para que sea tan grande como los otros.
* `[UNK] (desconocido)`: Para representar palabras desconocidas.

Siguiendo el ejemplo, habiendo tokenizado un texto asignando a cada palabra y símbolo del texto una posición en el vocabulario, la oración tokenizada `"Hello, world!"` -> `["Hello", ",", "world", "!"]` sería algo como: `[64, 455, 78, 467]` suponiendo que `Hello` está en la posición 64, "`,"` está en la posición `455`... en el array de vocabulario resultante.

Sin embargo, si en el texto utilizado para generar el vocabulario la palabra `"Bye"` no existía, esto resultará en: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` suponiendo que el token para `[UNK]` está en 987.

### BPE - Byte Pair Encoding

Para evitar problemas como la necesidad de tokenizar todas las posibles palabras para textos, los LLMs como GPT utilizan BPE que básicamente **codifica pares de bytes frecuentes** para reducir el tamaño del texto en un formato más optimizado hasta que no se pueda reducir más (ver [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Ten en cuenta que de esta manera no hay palabras "desconocidas" para el vocabulario y el vocabulario final será todos los conjuntos descubiertos de bytes frecuentes agrupados tanto como sea posible, mientras que los bytes que no están frecuentemente vinculados con el mismo byte serán un token por sí mismos.

## Data Sampling

Los LLMs como GPT funcionan prediciendo la siguiente palabra en función de las anteriores, por lo tanto, para preparar algunos datos para el entrenamiento es necesario preparar los datos de esta manera.

Por ejemplo, usando el texto "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Para preparar el modelo para aprender a predecir la siguiente palabra (suponiendo que cada palabra es un token usando el tokenizador muy básico), y usando un tamaño máximo de 4 y una ventana deslizante de 1, así es como se debe preparar el texto:
```javascript
Input: [
["Lorem", "ipsum", "dolor", "sit"],
["ipsum", "dolor", "sit", "amet,"],
["dolor", "sit", "amet,", "consectetur"],
["sit", "amet,", "consectetur", "adipiscing"],
],
Target: [
["ipsum", "dolor", "sit", "amet,"],
["dolor", "sit", "amet,", "consectetur"],
["sit", "amet,", "consectetur", "adipiscing"],
["amet,", "consectetur", "adipiscing", "elit,"],
["consectetur", "adipiscing", "elit,", "sed"],
]
```
Nota que si la ventana deslizante hubiera sido 2, significa que la siguiente entrada en el array de entrada comenzará 2 tokens después y no solo uno, pero el array objetivo seguirá prediciendo solo 1 token. En pytorch, esta ventana deslizante se expresa en el parámetro `stride`.
