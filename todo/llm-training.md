# LLM Training

## Tokenizing

Tokenizing bestaan uit die skeiding van die data in spesifieke stukke en die toekenning van spesifieke ID's (nommers) aan hulle.\
'n Baie eenvoudige tokenizer vir teks kan net elke woord van 'n teks apart neem, sowel as leestekens en spaties verwyder.\
Daarom, `"Hello, world!"` sou wees: `["Hello", ",", "world", "!"]`

Dan, om elkeen van die woorde en simbole 'n token ID (nommer) toe te ken, is dit nodig om die tokenizer **vokabulêre** te skep. As jy byvoorbeeld 'n boek tokeniseer, kan dit **al die verskillende woorde van die boek** in alfabetiese volgorde wees met 'n paar ekstra tokens soos:

* `[BOS] (Begin van reeks)`: Geplaas aan die begin van 'n teks, dui dit die begin van 'n teks aan (gebruik om nie-verwante tekste te skei).
* `[EOS] (Einde van reeks)`: Geplaas aan die einde van 'n teks, dui dit die einde van 'n teks aan (gebruik om nie-verwante tekste te skei).
* `[PAD] (padding)`: Wanneer 'n batchgrootte groter is as een (gewoonlik), word hierdie token gebruik om die lengte van daardie batch te verhoog om so groot soos die ander te wees.
* `[UNK] (onbekend)`: Om onbekende woorde te verteenwoordig.

Volg die voorbeeld, nadat 'n teks getokeniseer is deur elke woord en simbool van die teks 'n posisie in die vokabulêre toe te ken, sou die getokeniseerde sin `"Hello, world!"` -> `["Hello", ",", "world", "!"]` iets soos wees: `[64, 455, 78, 467]` met die aanname dat `Hello` op pos 64 is, "`,"` is op pos `455`... in die resulterende vokabulêre array.

As die woord `"Bye"` egter nie in die teks wat gebruik is om die vokabulêre te genereer bestaan het nie, sal dit lei tot: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` met die aanname dat die token vir `[UNK]` op 987 is.

### BPE - Byte Pair Encoding

Om probleme te vermy soos die behoefte om al die moontlike woorde vir teks te tokeniseer, gebruik LLMs soos GPT BPE wat basies **frekwente pare van bytes kodeer** om die grootte van die teks in 'n meer geoptimaliseerde formaat te verminder totdat dit nie verder verminder kan word nie (kyk [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Let daarop dat daar op hierdie manier geen "onbekende" woorde vir die vokabulêre is nie en die finale vokabulêre sal al die ontdekte stelle van frekwente bytes wees wat saam gegroepeer is so veel as moontlik terwyl bytes wat nie gereeld met dieselfde byte gekoppel is nie, 'n token op hul eie sal wees.

## Data Sampling

LLMs soos GPT werk deur die volgende woord te voorspel op grond van die vorige, daarom is dit nodig om die data op hierdie manier voor te berei vir opleiding.

Byvoorbeeld, met die teks "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Om die model voor te berei om die volgende woord te leer voorspel (met die aanname dat elke woord 'n token is wat die baie basiese tokenizer gebruik), en met 'n maksimum grootte van 4 en 'n skuifvenster van 1, is dit hoe die teks voorberei moet word:
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
Let wel dat as die skuifvenster 2 sou wees, dit beteken dat die volgende invoer in die invoerarray 2 tokens na die huidige een sal begin en nie net een nie, maar die teikenarray sal steeds net 1 token voorspel. In pytorch word hierdie skuifvenster in die parameter `stride` uitgedruk.
