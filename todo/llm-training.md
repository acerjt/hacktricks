# LLM Training

## Tokenizing

Tokenizing consiste nel separare i dati in specifici chunk e assegnare loro ID specifici (numeri).\
Un tokenizer molto semplice per i testi potrebbe semplicemente prendere ogni parola di un testo separatamente, e anche i simboli di punteggiatura e rimuovere gli spazi.\
Pertanto, `"Hello, world!"` sarebbe: `["Hello", ",", "world", "!"]`

Poi, per assegnare a ciascuna delle parole e simboli un token ID (numero), è necessario creare il **vocabolario** del tokenizer. Se stai tokenizzando, ad esempio, un libro, questo potrebbe essere **tutte le diverse parole del libro** in ordine alfabetico con alcuni token extra come:

* `[BOS] (Inizio della sequenza)`: Posizionato all'inizio di un testo, indica l'inizio di un testo (usato per separare testi non correlati).
* `[EOS] (Fine della sequenza)`: Posizionato alla fine di un testo, indica la fine di un testo (usato per separare testi non correlati).
* `[PAD] (padding)`: Quando una dimensione del batch è maggiore di uno (di solito), questo token è usato per aumentare la lunghezza di quel batch per essere grande come gli altri.
* `[UNK] (sconosciuto)`: Per rappresentare parole sconosciute.

Seguendo l'esempio, avendo tokenizzato un testo assegnando a ciascuna parola e simbolo del testo una posizione nel vocabolario, la frase tokenizzata `"Hello, world!"` -> `["Hello", ",", "world", "!"]` sarebbe qualcosa come: `[64, 455, 78, 467]` supponendo che `Hello` sia alla posizione 64, "`,"` sia alla posizione `455`... nell'array del vocabolario risultante.

Tuttavia, se nel testo usato per generare il vocabolario la parola `"Bye"` non esistesse, questo risulterebbe in: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` supponendo che il token per `[UNK]` sia a 987.

### BPE - Byte Pair Encoding

Per evitare problemi come la necessità di tokenizzare tutte le possibili parole per i testi, i LLM come GPT utilizzano BPE che fondamentalmente **codifica coppie di byte frequenti** per ridurre la dimensione del testo in un formato più ottimizzato fino a quando non può essere ridotto ulteriormente (controlla [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Nota che in questo modo non ci sono parole "sconosciute" per il vocabolario e il vocabolario finale sarà tutti i set scoperti di byte frequenti raggruppati il più possibile mentre i byte che non sono frequentemente collegati con lo stesso byte saranno un token a sé.

## Data Sampling

I LLM come GPT funzionano prevedendo la parola successiva basata su quelle precedenti, quindi per preparare alcuni dati per l'addestramento è necessario preparare i dati in questo modo.

Ad esempio, usando il testo "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Per preparare il modello ad apprendere a prevedere la parola successiva (supponendo che ogni parola sia un token usando il tokenizer molto basilare), e usando una dimensione massima di 4 e una finestra mobile di 1, questo è come il testo dovrebbe essere preparato:
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
Nota che se la finestra scorrevole fosse stata 2, significa che la prossima voce nell'array di input inizierà 2 token dopo e non solo uno, ma l'array target continuerà a prevedere solo 1 token. In pytorch, questa finestra scorrevole è espressa nel parametro `stride`.
