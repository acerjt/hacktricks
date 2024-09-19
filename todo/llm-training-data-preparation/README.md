# LLM Training - Data Preparation

**Це мої нотатки з дуже рекомендованої книги** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **з деякою додатковою інформацією.**

## Basic Information

Вам слід почати з читання цього посту для деяких базових концепцій, які ви повинні знати:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Мета цього початкового етапу дуже проста: **Розділіть вхідні дані на токени (ідентифікатори) таким чином, щоб це мало сенс**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Мета цього другого етапу дуже проста: **Вибірка вхідних даних і підготовка їх до етапу навчання, зазвичай шляхом розділення набору даних на речення певної довжини та також генерування очікуваної відповіді.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Мета цього третього етапу дуже проста: **Призначити кожному з попередніх токенів у словнику вектор бажаних розмірів для навчання моделі.** Кожне слово в словнику буде точкою в просторі X вимірів.\
Зверніть увагу, що спочатку позиція кожного слова в просторі просто ініціалізується "випадковим чином", і ці позиції є навчальними параметрами (будуть покращені під час навчання).

Більше того, під час вбудовування токенів **створюється ще один шар вбудовувань**, який представляє (в цьому випадку) **абсолютну позицію слова в навчальному реченні**. Таким чином, слово в різних позиціях у реченні матиме різне представлення (значення).
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Мета цього четвертого етапу дуже проста: **Застосувати деякі механізми уваги**. Це будуть багато **повторюваних шарів**, які будуть **фіксувати зв'язок слова в словнику з його сусідами в поточному реченні, що використовується для навчання LLM**.\
Для цього використовується багато шарів, тому багато навчальних параметрів будуть фіксувати цю інформацію.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Мета цього п'ятого етапу дуже проста: **Розробити архітектуру повного LLM**. З'єднайте все разом, застосуйте всі шари та створіть усі функції для генерації тексту або перетворення тексту в ідентифікатори і назад.

Ця архітектура буде використовуватися як для навчання, так і для прогнозування тексту після його навчання.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Мета цього шостого етапу дуже проста: **Навчити модель з нуля**. Для цього буде використана попередня архітектура LLM з деякими циклами, що проходять через набори даних, використовуючи визначені функції втрат і оптимізатор для навчання всіх параметрів моделі.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
Використання **LoRA значно зменшує обчислення**, необхідні для **тонкої настройки** вже навчених моделей.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Мета цього розділу - показати, як тонко налаштувати вже попередньо навчена модель, щоб замість генерації нового тексту LLM вибирав **ймовірності того, що даний текст буде класифікований у кожну з наданих категорій** (наприклад, чи є текст спамом чи ні).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Мета цього розділу - показати, як **тонко налаштувати вже попередньо навчена модель, щоб слідувати інструкціям**, а не просто генерувати текст, наприклад, відповідати на завдання як чат-бот.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
