# LLM Training - Data Preparation

**Hizi ni nota zangu kutoka kwa kitabu kinachopendekezwa sana** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **pamoja na taarifa za ziada.**

## Basic Information

Unapaswa kuanza kwa kusoma chapisho hili kwa baadhi ya dhana za msingi unazopaswa kujua kuhusu:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Lengo la awamu hii ya awali ni rahisi sana: **Gawanya ingizo katika tokeni (ids) kwa njia ambayo ina maana**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Lengo la awamu hii ya pili ni rahisi sana: **Chukua sampuli ya data ya ingizo na kuandaa kwa awamu ya mafunzo kwa kawaida kwa kutenganisha dataset katika sentensi za urefu maalum na pia kuzalisha jibu linalotarajiwa.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Lengo la awamu hii ya tatu ni rahisi sana: **Patia kila moja ya tokeni za awali katika msamiati vector ya vipimo vinavyotakiwa ili kufundisha mfano.** Kila neno katika msamiati litakuwa na pointi katika nafasi ya vipimo X.\
Kumbuka kwamba awali nafasi ya kila neno katika nafasi inaanzishwa "kwa bahati" na nafasi hizi ni vigezo vinavyoweza kufundishwa (vitaboreshwa wakati wa mafunzo).

Zaidi ya hayo, wakati wa kuingiza token **tabaka lingine la kuingiza linaundwa** ambalo linawakilisha (katika kesi hii) **nafasi halisi ya neno katika sentensi ya mafunzo**. Kwa njia hii neno katika nafasi tofauti katika sentensi litakuwa na uwakilishi tofauti (maana).
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Lengo la awamu hii ya nne ni rahisi sana: **Tumia baadhi ya mitambo ya umakini**. Hizi zitakuwa tabaka nyingi **zinazorudiwa** ambazo zitakuwa **zinakamata uhusiano wa neno katika msamiati na majirani zake katika sentensi ya sasa inayotumika kufundisha LLM**.\
Tabaka nyingi zinatumika kwa hili, hivyo vigezo vingi vinavyoweza kufundishwa vitakuwa vinakamata taarifa hii.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Lengo la awamu hii ya tano ni rahisi sana: **Tengeneza muundo wa LLM kamili**. Panga kila kitu pamoja, tumia tabaka zote na uunde kazi zote za kuzalisha maandiko au kubadilisha maandiko kuwa IDs na kinyume chake.

Muundo huu utatumika kwa mafunzo na kutabiri maandiko baada ya kufundishwa.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Lengo la awamu hii ya sita ni rahisi sana: **Fundisha mfano kutoka mwanzo**. Kwa hili muundo wa awali wa LLM utatumika na mizunguko kadhaa ikipita juu ya seti za data kwa kutumia kazi za hasara zilizofafanuliwa na optimizer ili kufundisha vigezo vyote vya mfano.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
Matumizi ya **LoRA hupunguza sana hesabu** inayohitajika ili **kurekebisha** mifano iliyofundishwa tayari.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Lengo la sehemu hii ni kuonyesha jinsi ya kurekebisha mfano uliofundishwa tayari ili badala ya kuzalisha maandiko mapya LLM itachagua kutoa **uwezekano wa maandiko yaliyotolewa kuainishwa katika kila moja ya makundi yaliyotolewa** (kama maandiko ni spam au la).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Lengo la sehemu hii ni kuonyesha jinsi ya **kurekebisha mfano uliofundishwa tayari ili kufuata maagizo** badala ya tu kuzalisha maandiko, kwa mfano, kujibu kazi kama roboti ya mazungumzo.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
