# LLM Training

## Tokenizing

Tokenizing besteht darin, die Daten in spezifische Teile zu trennen und ihnen spezifische IDs (Zahlen) zuzuweisen.\
Ein sehr einfacher Tokenizer für Texte könnte einfach jedes Wort eines Textes separat erfassen, sowie Interpunktionssymbole und Leerzeichen entfernen.\
Daher würde `"Hello, world!"` so aussehen: `["Hello", ",", "world", "!"]`

Um dann jedem der Wörter und Symbole eine Token-ID (Zahl) zuzuweisen, ist es notwendig, das **Wortschatz** des Tokenizers zu erstellen. Wenn Sie beispielsweise ein Buch tokenisieren, könnte dies **alle verschiedenen Wörter des Buches** in alphabetischer Reihenfolge mit einigen zusätzlichen Tokens wie:

* `[BOS] (Beginn der Sequenz)`: Am Anfang eines Textes platziert, zeigt es den Beginn eines Textes an (wird verwendet, um nicht verwandte Texte zu trennen).
* `[EOS] (Ende der Sequenz)`: Am Ende eines Textes platziert, zeigt es das Ende eines Textes an (wird verwendet, um nicht verwandte Texte zu trennen).
* `[PAD] (Padding)`: Wenn die Batch-Größe größer als eins ist (normalerweise), wird dieses Token verwendet, um die Länge dieser Batch so zu erhöhen, dass sie so groß wie die anderen ist.
* `[UNK] (unbekannt)`: Um unbekannte Wörter darzustellen.

Folgendes Beispiel, nachdem ein Text tokenisiert wurde, indem jedem Wort und Symbol des Textes eine Position im Wortschatz zugewiesen wurde, würde der tokenisierte Satz `"Hello, world!"` -> `["Hello", ",", "world", "!"]` so aussehen: `[64, 455, 78, 467]`, vorausgesetzt, dass `Hello` an Pos 64 ist, "`,"` an Pos `455`... im resultierenden Wortschatz-Array.

Wenn jedoch im Text, der zur Erstellung des Wortschatzes verwendet wurde, das Wort `"Bye"` nicht existierte, würde dies zu folgendem Ergebnis führen: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]`, vorausgesetzt, das Token für `[UNK]` ist an 987.

### BPE - Byte Pair Encoding

Um Probleme wie die Notwendigkeit zu vermeiden, alle möglichen Wörter für Texte zu tokenisieren, verwenden LLMs wie GPT BPE, das im Wesentlichen **häufige Byte-Paare kodiert**, um die Größe des Textes in einem optimierteren Format zu reduzieren, bis sie nicht weiter reduziert werden kann (siehe [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Beachten Sie, dass es auf diese Weise keine "unbekannten" Wörter für den Wortschatz gibt und der endgültige Wortschatz alle entdeckten Mengen häufiger Bytes zusammen gruppiert, während Bytes, die nicht häufig mit demselben Byte verknüpft sind, ein Token für sich selbst sein werden.

## Data Sampling

LLMs wie GPT arbeiten, indem sie das nächste Wort basierend auf den vorherigen vorhersagen. Daher ist es notwendig, die Daten auf diese Weise vorzubereiten.

Zum Beispiel, unter Verwendung des Textes "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Um das Modell darauf vorzubereiten, das folgende Wort vorherzusagen (vorausgesetzt, jedes Wort ist ein Token, das den sehr einfachen Tokenizer verwendet), und unter Verwendung einer maximalen Größe von 4 und einem gleitenden Fenster von 1, sollte der Text so vorbereitet werden:
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
Beachten Sie, dass wenn das gleitende Fenster 2 gewesen wäre, dies bedeutet, dass der nächste Eintrag im Eingabearray 2 Tokens später beginnt und nicht nur einen, aber das Zielarray wird weiterhin nur 1 Token vorhersagen. In pytorch wird dieses gleitende Fenster im Parameter `stride` ausgedrückt.
