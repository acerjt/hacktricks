# LLM Training

## Tokenizing

Tokenizing se sastoji od odvajanja podataka u specifične delove i dodeljivanja specifičnih ID-ova (brojeva).\
Veoma jednostavan tokenizer za tekstove može da uzme svaku reč iz teksta posebno, kao i interpunkcijske simbole i ukloni razmake.\
Dakle, `"Hello, world!"` bi bilo: `["Hello", ",", "world", "!"]`

Zatim, da bi se svakoj od reči i simbola dodelio token ID (broj), potrebno je kreirati **rečnik** tokenizera. Ako tokenizujete, na primer, knjigu, to bi moglo biti **sve različite reči iz knjige** u abecednom redu sa nekim dodatnim tokenima kao što su:

* `[BOS] (Početak sekvence)`: Postavljen na početku teksta, označava početak teksta (koristi se za odvajanje nepovezanih tekstova).
* `[EOS] (Kraj sekvence)`: Postavljen na kraju teksta, označava kraj teksta (koristi se za odvajanje nepovezanih tekstova).
* `[PAD] (podešavanje)`: Kada je veličina serije veća od jedan (obično), ovaj token se koristi da poveća dužinu te serije da bude što veća kao ostale.
* `[UNK] (nepoznato)`: Da predstavlja nepoznate reči.

Sledeći primer, nakon tokenizacije teksta dodeljujući svakoj reči i simbolu iz teksta poziciju u rečniku, tokenizovana rečenica `"Hello, world!"` -> `["Hello", ",", "world", "!"]` bi bila nešto poput: `[64, 455, 78, 467]` pretpostavljajući da je `Hello` na poziciji 64, "`,"` je na poziciji `455`... u rezultantnom nizu rečnika.

Međutim, ako u tekstu korišćenom za generisanje rečnika reč `"Bye"` nije postojala, to će rezultirati: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` pretpostavljajući da je token za `[UNK]` na 987.

### BPE - Byte Pair Encoding

Da bi se izbegli problemi poput potrebe da se tokenizuju sve moguće reči za tekstove, LLM-ovi poput GPT koriste BPE koji suštinski **kodira česte parove bajtova** kako bi smanjio veličinu teksta u optimizovanijem formatu dok se ne može više smanjiti (proverite [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Imajte na umu da na ovaj način nema "nepoznatih" reči za rečnik i konačni rečnik će biti svi otkriveni setovi čestih bajtova grupisani što je više moguće dok bajtovi koji nisu često povezani sa istim bajtom će biti token sami po sebi.

## Data Sampling

LLM-ovi poput GPT rade predviđajući sledeću reč na osnovu prethodnih, stoga je neophodno pripremiti podatke za obuku na ovaj način.

Na primer, koristeći tekst "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Da bi se model pripremio da uči predviđajući sledeću reč (pretpostavljajući da je svaka reč token koristeći veoma osnovni tokenizer), i koristeći maksimalnu veličinu od 4 i klizni prozor od 1, ovako bi tekst trebao biti pripremljen:
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
Napomena da, ako bi klizni prozor bio 2, to znači da će sledeći unos u ulaznom nizu početi 2 tokena kasnije, a ne samo jedan, ali će ciljni niz i dalje predviđati samo 1 token. U pytorchu, ovaj klizni prozor se izražava u parametru `stride`.
