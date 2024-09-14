# LLM Training

## Tokenizing

Tokenizing consiste em separar os dados em pedaços específicos e atribuir a eles IDs específicos (números).\
Um tokenizador muito simples para textos pode simplesmente pegar cada palavra de um texto separadamente, e também símbolos de pontuação e remover espaços.\
Portanto, `"Hello, world!"` seria: `["Hello", ",", "world", "!"]`

Então, para atribuir a cada uma das palavras e símbolos um ID de token (número), é necessário criar o **vocabulário** do tokenizador. Se você estiver tokenizando, por exemplo, um livro, isso poderia ser **todas as diferentes palavras do livro** em ordem alfabética com alguns tokens extras como:

* `[BOS] (Início da sequência)`: Colocado no início de um texto, indica o começo de um texto (usado para separar textos não relacionados).
* `[EOS] (Fim da sequência)`: Colocado no final de um texto, indica o fim de um texto (usado para separar textos não relacionados).
* `[PAD] (preenchimento)`: Quando o tamanho do lote é maior que um (geralmente), este token é usado para aumentar o comprimento desse lote para ser tão grande quanto os outros.
* `[UNK] (desconhecido)`: Para representar palavras desconhecidas.

Seguindo o exemplo, tendo tokenizado um texto atribuindo a cada palavra e símbolo do texto uma posição no vocabulário, a frase tokenizada `"Hello, world!"` -> `["Hello", ",", "world", "!"]` seria algo como: `[64, 455, 78, 467]` supondo que `Hello` está na posição 64, "`,"` está na posição `455`... no array de vocabulário resultante.

No entanto, se no texto usado para gerar o vocabulário a palavra `"Bye"` não existisse, isso resultaria em: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` supondo que o token para `[UNK]` está na posição 987.

### BPE - Byte Pair Encoding

Para evitar problemas como a necessidade de tokenizar todas as palavras possíveis para textos, LLMs como GPT usam BPE que basicamente **codifica pares frequentes de bytes** para reduzir o tamanho do texto em um formato mais otimizado até que não possa ser reduzido mais (ver [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Note que dessa forma não há palavras "desconhecidas" para o vocabulário e o vocabulário final será todos os conjuntos descobertos de bytes frequentes agrupados o máximo possível, enquanto bytes que não estão frequentemente ligados ao mesmo byte serão um token por si mesmos.

## Data Sampling

LLMs como GPT funcionam prevendo a próxima palavra com base nas anteriores, portanto, para preparar alguns dados para treinamento, é necessário preparar os dados dessa forma.

Por exemplo, usando o texto "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Para preparar o modelo para aprender a prever a próxima palavra (supondo que cada palavra é um token usando o tokenizador muito básico), e usando um tamanho máximo de 4 e uma janela deslizante de 1, é assim que o texto deve ser preparado:
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
Note que se a janela deslizante fosse 2, isso significa que a próxima entrada no array de entrada começará 2 tokens depois e não apenas um, mas o array alvo ainda estará prevendo apenas 1 token. No pytorch, essa janela deslizante é expressa no parâmetro `stride`.
