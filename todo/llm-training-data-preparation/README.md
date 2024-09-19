# LLM Training - Data Preparation

**ये मेरी नोट्स हैं बहुत ही अनुशंसित किताब से** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **कुछ अतिरिक्त जानकारी के साथ।**

## Basic Information

आपको कुछ बुनियादी अवधारणाओं के बारे में जानने के लिए इस पोस्ट को पढ़ना चाहिए:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
इस प्रारंभिक चरण का लक्ष्य बहुत सरल है: **इनपुट को कुछ इस तरह से टोकन (ids) में विभाजित करें जो समझ में आए।**
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
इस दूसरे चरण का लक्ष्य बहुत सरल है: **इनपुट डेटा का सैंपल लें और इसे प्रशिक्षण चरण के लिए तैयार करें, आमतौर पर डेटासेट को एक विशिष्ट लंबाई के वाक्यों में विभाजित करके और अपेक्षित प्रतिक्रिया भी उत्पन्न करके।**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
इस तीसरे चरण का लक्ष्य बहुत सरल है: **शब्दकोश में पिछले टोकनों में से प्रत्येक को मॉडल को प्रशिक्षित करने के लिए इच्छित आयामों का एक वेक्टर सौंपें।** शब्दकोश में प्रत्येक शब्द X आयामों के एक स्थान में एक बिंदु होगा।\
ध्यान दें कि प्रारंभ में प्रत्येक शब्द की स्थिति "यादृच्छिक रूप से" प्रारंभ की जाती है और ये स्थितियाँ प्रशिक्षित करने योग्य पैरामीटर हैं (प्रशिक्षण के दौरान सुधारित होंगी)।

इसके अलावा, टोकन एम्बेडिंग के दौरान **एक और एम्बेडिंग परत बनाई जाती है** जो (इस मामले में) **प्रशिक्षण वाक्य में शब्द की सटीक स्थिति का प्रतिनिधित्व करती है।** इस तरह, वाक्य में विभिन्न स्थितियों में एक शब्द का अलग प्रतिनिधित्व (अर्थ) होगा।
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
इस चौथे चरण का लक्ष्य बहुत सरल है: **कुछ ध्यान तंत्र लागू करें।** ये बहुत सारे **दोहराए जाने वाले परतें** होंगी जो **शब्दकोश में एक शब्द के पड़ोसियों के साथ वर्तमान वाक्य में संबंध को कैप्चर करेंगी जिसका उपयोग LLM को प्रशिक्षित करने के लिए किया जा रहा है।**\
इसके लिए बहुत सारी परतें उपयोग की जाती हैं, इसलिए बहुत सारे प्रशिक्षित करने योग्य पैरामीटर इस जानकारी को कैप्चर करने जा रहे हैं।
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
इस पांचवे चरण का लक्ष्य बहुत सरल है: **पूर्ण LLM की आर्किटेक्चर विकसित करें।** सब कुछ एक साथ रखें, सभी परतें लागू करें और पाठ उत्पन्न करने या पाठ को IDs में और इसके विपरीत परिवर्तित करने के लिए सभी कार्यों को बनाएं।

यह आर्किटेक्चर दोनों के लिए उपयोग की जाएगी, प्रशिक्षण और भविष्यवाणी के लिए पाठ के बाद इसे प्रशिक्षित किया गया।
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
इस छठे चरण का लक्ष्य बहुत सरल है: **मॉडल को शून्य से प्रशिक्षित करें।** इसके लिए पिछले LLM आर्किटेक्चर का उपयोग किया जाएगा जिसमें डेटा सेट पर परिभाषित हानि कार्यों और ऑप्टिमाइज़र का उपयोग करते हुए लूप होंगे ताकि मॉडल के सभी पैरामीटर को प्रशिक्षित किया जा सके।
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
**LoRA का उपयोग पहले से प्रशिक्षित मॉडलों को **फाइन ट्यून** करने के लिए आवश्यक गणना को बहुत कम करता है।**
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
इस अनुभाग का लक्ष्य यह दिखाना है कि पहले से प्रशिक्षित मॉडल को कैसे फाइन-ट्यून किया जाए ताकि नया पाठ उत्पन्न करने के बजाय LLM **प्रत्येक दिए गए श्रेणी में वर्गीकृत होने के लिए दिए गए पाठ की संभावनाएँ** प्रदान करे (जैसे कि कोई पाठ स्पैम है या नहीं)।
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
इस अनुभाग का लक्ष्य यह दिखाना है कि **निर्देशों का पालन करने के लिए पहले से प्रशिक्षित मॉडल को कैसे फाइन-ट्यून किया जाए** न कि केवल पाठ उत्पन्न करने के लिए, उदाहरण के लिए, एक चैट बॉट के रूप में कार्यों का उत्तर देना।
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
