# LLM Training

## Tokenizing

Tokenizing polega na oddzieleniu danych na konkretne fragmenty i przypisaniu im specyficznych identyfikatorów (numerów).\
Bardzo prosty tokenizer dla tekstów może po prostu wziąć każde słowo z tekstu osobno, a także symbole interpunkcyjne i usunąć spacje.\
Dlatego `"Hello, world!"` byłoby: `["Hello", ",", "world", "!"]`

Następnie, aby przypisać każdemu z słów i symboli identyfikator tokena (numer), konieczne jest stworzenie **słownika** tokenizera. Jeśli tokenizujesz na przykład książkę, może to być **wszystkie różne słowa książki** w porządku alfabetycznym z dodatkowymi tokenami, takimi jak:

* `[BOS] (Początek sekwencji)`: Umieszczony na początku tekstu, wskazuje na początek tekstu (używany do oddzielania niepowiązanych tekstów).
* `[EOS] (Koniec sekwencji)`: Umieszczony na końcu tekstu, wskazuje na koniec tekstu (używany do oddzielania niepowiązanych tekstów).
* `[PAD] (wypełnienie)`: Gdy rozmiar partii jest większy niż jeden (zwykle), ten token jest używany do zwiększenia długości tej partii, aby była tak duża jak inne.
* `[UNK] (nieznane)`: Aby reprezentować nieznane słowa.

Podążając za przykładem, mając tokenizowany tekst, przypisując każdemu słowu i symbolowi tekstu pozycję w słowniku, tokenizowane zdanie `"Hello, world!"` -> `["Hello", ",", "world", "!"]` byłoby czymś takim: `[64, 455, 78, 467]`, zakładając, że `Hello` jest na pozycji 64, "`,"` jest na pozycji `455`... w wynikowej tablicy słownika.

Jednakże, jeśli w tekście użytym do wygenerowania słownika słowo `"Bye"` nie istniało, skutkowałoby to: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]`, zakładając, że token dla `[UNK]` jest na 987.

### BPE - Byte Pair Encoding

Aby uniknąć problemów, takich jak konieczność tokenizowania wszystkich możliwych słów dla tekstów, LLM-y takie jak GPT używają BPE, które zasadniczo **koduje częste pary bajtów**, aby zredukować rozmiar tekstu w bardziej zoptymalizowanym formacie, aż nie można go już bardziej zredukować (sprawdź [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Zauważ, że w ten sposób nie ma "nieznanych" słów dla słownika, a ostateczny słownik będzie zawierał wszystkie odkryte zestawy częstych bajtów, grupowane tak blisko, jak to możliwe, podczas gdy bajty, które nie są często powiązane z tym samym bajtem, będą tokenem samym w sobie.

## Data Sampling

LLM-y takie jak GPT działają, przewidując następne słowo na podstawie poprzednich, dlatego aby przygotować dane do treningu, konieczne jest przygotowanie danych w ten sposób.

Na przykład, używając tekstu "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Aby przygotować model do nauki przewidywania następnego słowa (zakładając, że każde słowo jest tokenem przy użyciu bardzo podstawowego tokenizera), i używając maksymalnego rozmiaru 4 oraz przesuwającego okna 1, oto jak tekst powinien być przygotowany:
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
Zauważ, że jeśli okno przesuwne wynosiłoby 2, oznacza to, że następny wpis w tablicy wejściowej zacznie się 2 tokeny później, a nie tylko jeden, ale tablica docelowa nadal będzie przewidywać tylko 1 token. W pytorch to okno przesuwne jest wyrażone w parametrze `stride`.
