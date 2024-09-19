# LLM Training - Data Preparation

**Ovo su moje beleške iz veoma preporučene knjige** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **sa dodatnim informacijama.**

## Basic Information

Trebalo bi da počnete čitanjem ovog posta za neke osnovne koncepte koje treba da znate:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Cilj ove inicijalne faze je veoma jednostavan: **Podeliti ulaz u tokene (ids) na način koji ima smisla**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Cilj ove druge faze je veoma jednostavan: **Uzmite uzorak ulaznih podataka i pripremite ih za fazu obuke obično razdvajanjem skupa podataka na rečenice određene dužine i generisanjem očekivanog odgovora.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Cilj ove treće faze je veoma jednostavan: **Dodeliti svakom od prethodnih tokena u rečniku vektor željenih dimenzija za obuku modela.** Svaka reč u rečniku će biti tačka u prostoru X dimenzija.\
Napomena: inicijalno je pozicija svake reči u prostoru "nasumično" inicijalizovana i te pozicije su parametri koji se mogu obučavati (biće poboljšani tokom obuke).

Pored toga, tokom token embedding **stvara se još jedan sloj embedding-a** koji predstavlja (u ovom slučaju) **apsolutnu poziciju reči u rečenici za obuku**. Na ovaj način, reč na različitim pozicijama u rečenici će imati različitu reprezentaciju (značenje).
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Cilj ove četvrte faze je veoma jednostavan: **Primena nekih mehanizama pažnje**. Ovi mehanizmi će biti mnogo **ponovljenih slojeva** koji će **uhvatiti odnos reči u rečniku sa njenim susedima u trenutnoj rečenici koja se koristi za obuku LLM-a**.\
Za ovo se koristi mnogo slojeva, tako da će mnogo parametara koji se mogu obučavati uhvatiti ove informacije.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Cilj ove pete faze je veoma jednostavan: **Razviti arhitekturu celog LLM-a**. Spojiti sve, primeniti sve slojeve i kreirati sve funkcije za generisanje teksta ili transformaciju teksta u ID-ove i obrnuto.

Ova arhitektura će se koristiti i za obuku i za predikciju teksta nakon što je obučena.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Cilj ove šeste faze je veoma jednostavan: **Obučiti model od nule**. Za ovo će se koristiti prethodna LLM arhitektura sa nekim petljama koje prolaze kroz skupove podataka koristeći definisane funkcije gubitka i optimizator za obuku svih parametara modela.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
Korišćenje **LoRA značajno smanjuje računarske resurse** potrebne za **fino podešavanje** već obučenih modela.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Cilj ovog odeljka je pokazati kako fino podešavati već obučeni model tako da umesto generisanja novog teksta LLM daje **verovatnoće da dati tekst bude kategorizovan u svaku od datih kategorija** (kao što je da li je tekst spam ili ne).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Cilj ovog odeljka je pokazati kako **fino podešavati već obučeni model da prati uputstva** umesto samo generisanja teksta, na primer, odgovaranje na zadatke kao chat bot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
