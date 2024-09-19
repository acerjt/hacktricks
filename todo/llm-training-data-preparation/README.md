# LLM Training - Data Preparation

**To są moje notatki z bardzo polecanej książki** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **z dodatkowymi informacjami.**

## Basic Information

Powinieneś zacząć od przeczytania tego posta, aby poznać podstawowe pojęcia, które powinieneś znać:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Celem tej początkowej fazy jest bardzo proste: **Podzielić dane wejściowe na tokeny (id) w sposób, który ma sens**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Celem tej drugiej fazy jest bardzo proste: **Próbkować dane wejściowe i przygotować je do fazy treningowej, zazwyczaj dzieląc zbiór danych na zdania o określonej długości i generując również oczekiwaną odpowiedź.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Celem tej trzeciej fazy jest bardzo proste: **Przypisać każdemu z poprzednich tokenów w słowniku wektor o pożądanych wymiarach do trenowania modelu.** Każde słowo w słowniku będzie punktem w przestrzeni o X wymiarach.\
Zauważ, że początkowo pozycja każdego słowa w przestrzeni jest po prostu inicjowana "losowo", a te pozycje są parametrami, które można trenować (będą poprawiane podczas treningu).

Ponadto, podczas osadzania tokenów **tworzona jest kolejna warstwa osadzeń**, która reprezentuje (w tym przypadku) **absolutną pozycję słowa w zdaniu treningowym**. W ten sposób słowo w różnych pozycjach w zdaniu będzie miało różne reprezentacje (znaczenia).
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Celem tej czwartej fazy jest bardzo proste: **Zastosować pewne mechanizmy uwagi**. Będą to liczne **powtarzające się warstwy**, które będą **uchwytywać relację słowa w słowniku z jego sąsiadami w bieżącym zdaniu używanym do trenowania LLM**.\
Do tego celu używa się wielu warstw, więc wiele parametrów do trenowania będzie uchwytywać te informacje.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Celem tej piątej fazy jest bardzo proste: **Opracować architekturę pełnego LLM**. Połączyć wszystko, zastosować wszystkie warstwy i stworzyć wszystkie funkcje do generowania tekstu lub przekształcania tekstu na ID i odwrotnie.

Ta architektura będzie używana zarówno do trenowania, jak i przewidywania tekstu po jego wytrenowaniu.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Celem tej szóstej fazy jest bardzo proste: **Wytrenować model od podstaw**. W tym celu zostanie użyta wcześniejsza architektura LLM z pewnymi pętlami przechodzącymi przez zbiory danych, używając zdefiniowanych funkcji straty i optymalizatora do trenowania wszystkich parametrów modelu.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
Użycie **LoRA znacznie redukuje obliczenia** potrzebne do **dostosowania** już wytrenowanych modeli.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Celem tej sekcji jest pokazanie, jak dostosować już wytrenowany model, aby zamiast generować nowy tekst, LLM podałby **prawdopodobieństwa, że dany tekst zostanie zaklasyfikowany w każdej z podanych kategorii** (na przykład, czy tekst jest spamem, czy nie).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Celem tej sekcji jest pokazanie, jak **dostosować już wytrenowany model do przestrzegania instrukcji** zamiast tylko generować tekst, na przykład, odpowiadając na zadania jako chatbot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
