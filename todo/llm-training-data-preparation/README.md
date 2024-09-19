# LLM Training - Data Preparation

**Bunlar, çok önerilen** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **kitabından aldığım notlar ve bazı ek bilgiler.**

## Basic Information

Bu temel kavramlar hakkında bilmeniz gereken bazı temel bilgiler için bu yazıyı okumaya başlamalısınız:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Bu ilk aşamanın amacı çok basit: **Girdiyi mantıklı bir şekilde token'lara (kimliklere) ayırmak**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Bu ikinci aşamanın amacı çok basit: **Girdi verilerini örneklemek ve genellikle belirli bir uzunluktaki cümlelere ayırarak ve beklenen yanıtı da üreterek eğitim aşamasına hazırlamak.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Bu üçüncü aşamanın amacı çok basit: **Sözlükteki önceki her token'a modelin eğitimi için istenen boyutlarda bir vektör atamak.** Sözlükteki her kelime, X boyutlu bir uzayda bir nokta olacaktır.\
Başlangıçta her kelimenin uzaydaki konumu "rastgele" başlatılır ve bu konumlar eğitilebilir parametrelerdir (eğitim sırasında geliştirilecektir).

Ayrıca, token embedding sırasında **başka bir embedding katmanı oluşturulur** ki bu, **eğitim cümlesindeki kelimenin mutlak konumunu** temsil eder. Bu şekilde, cümledeki farklı konumlarda bir kelime farklı bir temsil (anlam) alacaktır.
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Bu dördüncü aşamanın amacı çok basit: **Bazı dikkat mekanizmaları uygulamak**. Bunlar, **sözlükteki bir kelimenin, LLM'yi eğitmek için kullanılan mevcut cümledeki komşularıyla olan ilişkisini yakalayacak çok sayıda **tekrarlanan katman** olacaktır.\
Bunun için çok sayıda katman kullanılacak, bu nedenle çok sayıda eğitilebilir parametre bu bilgiyi yakalayacaktır.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Bu beşinci aşamanın amacı çok basit: **Tam LLM'nin mimarisini geliştirmek**. Her şeyi bir araya getirin, tüm katmanları uygulayın ve metin oluşturmak veya metni kimliklere ve geriye dönüştürmek için tüm işlevleri oluşturun.

Bu mimari, hem eğitim hem de eğitimden sonra metin tahmin etmek için kullanılacaktır.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Bu altıncı aşamanın amacı çok basit: **Modeli sıfırdan eğitmek**. Bunun için önceki LLM mimarisi, tanımlı kayıp fonksiyonları ve optimizasyon kullanarak veri setleri üzerinde döngülerle tüm model parametrelerini eğitmek için kullanılacaktır.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
**LoRA'nın kullanımı,** zaten eğitilmiş modelleri **ince ayar yapmak için gereken hesaplamayı** büyük ölçüde azaltır.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Bu bölümün amacı, LLM'nin yeni metin oluşturmak yerine **verilen metnin her bir verilen kategoriye sınıflandırılma olasılıklarını** seçmesini sağlamak için zaten önceden eğitilmiş bir modeli nasıl ince ayar yapacağınızı göstermektir (örneğin, bir metnin spam olup olmadığını belirlemek).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Bu bölümün amacı, sadece metin oluşturmak yerine **talimatları takip etmek için zaten önceden eğitilmiş bir modeli nasıl ince ayar yapacağınızı** göstermektir; örneğin, bir sohbet botu olarak görevlere yanıt vermek.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
