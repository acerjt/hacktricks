# LLM Training - Data Preparation

**Dit is my notas van die baie aanbevole boek** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **met 'n paar ekstra inligting.**

## Basic Information

Jy moet begin deur hierdie pos te lees vir 'n paar basiese konsepte wat jy moet weet oor:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Die doel van hierdie aanvanklike fase is baie eenvoudig: **Verdeel die invoer in tokens (ids) op 'n manier wat sin maak**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Die doel van hierdie tweede fase is baie eenvoudig: **Steek die invoerdata en berei dit voor vir die opleidingsfase deur gewoonlik die datastel in sinne van 'n spesifieke lengte te skei en ook die verwagte reaksie te genereer.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Die doel van hierdie derde fase is baie eenvoudig: **Ken elkeen van die vorige tokens in die woordeskat 'n vektor van die gewenste dimensies toe om die model te train.** Elke woord in die woordeskat sal 'n punt in 'n ruimte van X dimensies wees.\
Let daarop dat die posisie van elke woord in die ruimte aanvanklik net "ewekansig" geinitialiseer word en hierdie posisies is leerbare parameters (sal verbeter word tydens die opleiding).

Boonop, tydens die token embedding **word 'n ander laag van embeddings geskep** wat (in hierdie geval) die **absolute posisie van die woord in die opleidingssin** verteenwoordig. Op hierdie manier sal 'n woord in verskillende posisies in die sin 'n ander voorstelling (betekenis) hê.
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Die doel van hierdie vierde fase is baie eenvoudig: **Pas 'n paar aandagmeganismes toe**. Hierdie gaan baie **herhaalde lae** wees wat die **verhouding van 'n woord in die woordeskat met sy bure in die huidige sin wat gebruik word om die LLM te train, vasvang**.\
Daar word baie lae hiervoor gebruik, so baie leerbare parameters gaan hierdie inligting vasvang.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Die doel van hierdie vyfde fase is baie eenvoudig: **Ontwikkel die argitektuur van die volle LLM**. Sit alles saam, pas al die lae toe en skep al die funksies om teks te genereer of teks na IDs en terug te transformeer.

Hierdie argitektuur sal vir beide, opleiding en voorspellings van teks gebruik word nadat dit opgelei is.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Die doel van hierdie sesde fase is baie eenvoudig: **Oplei die model van nuuts af**. Hiervoor sal die vorige LLM-argitektuur gebruik word met 'n paar lusse wat oor die datastelle gaan met behulp van die gedefinieerde verliesfunksies en optimizer om al die parameters van die model op te lei.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
Die gebruik van **LoRA verminder baie die berekening** wat nodig is om **fine tune** reeds opgelei modelle.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Die doel van hierdie afdeling is om te wys hoe om 'n reeds vooropgeleide model te fine-tune sodat in plaas daarvan om nuwe teks te genereer, die LLM die **waarskynlikhede van die gegewe teks om in elkeen van die gegewe kategorieë gekategoriseer te word** (soos of 'n teks spam is of nie) sal gee.
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Die doel van hierdie afdeling is om te wys hoe om **'n reeds vooropgeleide model te fine-tune om instruksies te volg** eerder as net teks te genereer, byvoorbeeld, om op take te reageer as 'n chat bot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
