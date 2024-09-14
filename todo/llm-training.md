# LLM Training

## Tokenizing

Tokenizing inahusisha kutenganisha data katika vipande maalum na kupewa vitambulisho maalum (nambari).\
Tokenizer rahisi sana kwa maandiko inaweza kuwa tu kupata kila neno la maandiko kando, na pia alama za punctuation na kuondoa nafasi.\
Hivyo, `"Hello, world!"` itakuwa: `["Hello", ",", "world", "!"]`

Kisha, ili kupewa kila neno na alama ID ya token (nambari), inahitajika kuunda **vocabulary** ya tokenizer. Ikiwa unafanya tokenizing kwa mfano kitabu, hii inaweza kuwa **maneno yote tofauti ya kitabu** kwa mpangilio wa alfabeti na baadhi ya token za ziada kama:

* `[BOS] (Mwanzo wa mfuatano)`: Imewekwa mwanzoni mwa maandiko, inaashiria mwanzo wa maandiko (inayotumika kutenganisha maandiko yasiyo na uhusiano).
* `[EOS] (Mwisho wa mfuatano)`: Imewekwa mwishoni mwa maandiko, inaashiria mwisho wa maandiko (inayotumika kutenganisha maandiko yasiyo na uhusiano).
* `[PAD] (padding)`: Wakati saizi ya kundi ni kubwa kuliko moja (kawaida), token hii inatumika kuongeza urefu wa kundi hilo kuwa kubwa kama wengine.
* `[UNK] (haijulikani)`: Kuonyesha maneno yasiyojulikana.

Kufuata mfano, baada ya kufanya tokenizing maandiko kwa kupewa kila neno na alama ya maandiko nafasi katika vocabulary, sentensi iliyofanywa tokenized `"Hello, world!"` -> `["Hello", ",", "world", "!"]` itakuwa kitu kama: `[64, 455, 78, 467]` tukidhani kwamba `Hello` iko katika pos 64, "`,"` iko katika pos `455`... katika array ya vocabulary inayotokana.

Hata hivyo, ikiwa katika maandiko yaliyotumika kuunda vocabulary neno `"Bye"` halikuwapo, hii itasababisha: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` tukidhani token ya `[UNK]` iko katika 987.

### BPE - Byte Pair Encoding

Ili kuepuka matatizo kama vile kuhitaji kufanya tokenizing maneno yote yanayowezekana kwa maandiko, LLMs kama GPT hutumia BPE ambayo kimsingi **inaandika jozi za byte za mara kwa mara** ili kupunguza ukubwa wa maandiko katika muundo ulioimarishwa hadi haiwezi kupunguzwa zaidi (angalia [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Kumbuka kwamba kwa njia hii hakuna maneno "yasiyojulikana" kwa vocabulary na vocabulary ya mwisho itakuwa seti zote zilizogunduliwa za byte za mara kwa mara pamoja zimepangwa kadri iwezekanavyo wakati byte ambazo hazihusiani mara kwa mara na byte ile ile zitakuwa token wenyewe.

## Data Sampling

LLMs kama GPT hufanya kazi kwa kutabiri neno linalofuata kulingana na yale ya awali, hivyo ili kuandaa data fulani kwa mafunzo ni muhimu kuandaa data hii kwa njia hii.

Kwa mfano, kutumia maandiko "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Ili kuandaa mfano kujifunza kutabiri neno linalofuata (tukidhani kila neno ni token kwa kutumia tokenizer rahisi sana), na kutumia saizi ya juu ya 4 na dirisha linalosogea la 1, hii ndiyo jinsi maandiko yanapaswa kuandaliwa:
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
Kumbuka kwamba ikiwa dirisha linalosogea lingekuwa 2, inamaanisha kuwa kuingia kwa pili katika safu ya ingizo kutaanza tokeni 2 baada ya na si moja tu, lakini safu ya lengo bado itakuwa ikitabiri tokeni 1 tu. Katika pytorch, dirisha hili linalosogea linaonyeshwa katika parameter `stride`.
