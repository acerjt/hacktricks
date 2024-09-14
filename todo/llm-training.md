# LLM Training

## Tokenizing

Tokenizing, verileri belirli parçalara ayırmak ve onlara belirli kimlikler (numaralar) atamak anlamına gelir.\
Metinler için çok basit bir tokenizer, bir metindeki her kelimeyi ayrı ayrı almak ve ayrıca noktalama işaretlerini almak ve boşlukları kaldırmak olabilir.\
Bu nedenle, `"Hello, world!"` şöyle olur: `["Hello", ",", "world", "!"]`

Sonra, kelimelere ve sembollere bir token ID (numara) atamak için tokenizer **sözlüğü** oluşturmak gerekir. Örneğin, bir kitabı tokenize ediyorsanız, bu **kitabın tüm farklı kelimeleri** alfabetik sırayla ve bazı ekstra tokenlerle olabilir:

* `[BOS] (Dizinin başlangıcı)`: Bir metnin başında yer alır, bir metnin başlangıcını gösterir (ilişkisiz metinleri ayırmak için kullanılır).
* `[EOS] (Dizinin sonu)`: Bir metnin sonunda yer alır, bir metnin sonunu gösterir (ilişkisiz metinleri ayırmak için kullanılır).
* `[PAD] (doldurma)`: Bir batch boyutu genellikle birden büyük olduğunda, bu token, o batch'in uzunluğunu diğerleriyle aynı boyuta çıkarmak için kullanılır.
* `[UNK] (bilinmeyen)`: Bilinmeyen kelimeleri temsil etmek için.

Örneği takip ederek, bir metni tokenize edip her kelime ve sembole sözlükte bir pozisyon atadıktan sonra, tokenize edilmiş cümle `"Hello, world!"` -> `["Hello", ",", "world", "!"]` şöyle olur: `[64, 455, 78, 467]` varsayılarak `Hello` pozisyonda 64, "`,"` pozisyonda `455`... sonuçta oluşan sözlük dizisinde.

Ancak, sözlüğü oluşturmak için kullanılan metinde `"Bye"` kelimesi yoksa, bu sonuç verir: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` varsayılarak `[UNK]` için token 987'dir.

### BPE - Byte Pair Encoding

Tüm olası kelimeleri tokenize etme gibi sorunları önlemek için, GPT gibi LLM'ler, metnin boyutunu daha optimize bir formatta azaltmak için temelde **sık kullanılan byte çiftlerini kodlayan** BPE'yi kullanır, bu şekilde daha fazla azaltılamaz hale gelene kadar (bakınız [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Bu şekilde, sözlükte "bilinmeyen" kelimeler yoktur ve nihai sözlük, mümkün olduğunca gruplandırılmış sık kullanılan byte'ların keşfedilen tüm setleri olacaktır; aynı byte ile sıkça bağlantılı olmayan byte'lar ise kendileri bir token olacaktır.

## Data Sampling

GPT gibi LLM'ler, önceki kelimelere dayanarak bir sonraki kelimeyi tahmin ederek çalışır, bu nedenle eğitim için bazı verileri hazırlamak amacıyla verilerin bu şekilde hazırlanması gereklidir.

Örneğin, "Lorem ipsum dolor sit amet, consectetur adipiscing elit," metnini kullanarak

Modeli, bir sonraki kelimeyi tahmin etmeyi öğrenmeye hazırlamak için (her kelimenin çok temel bir tokenizer kullanarak bir token olduğunu varsayarak) ve maksimum boyut 4 ve kaydırma penceresi 1 kullanarak, metin şu şekilde hazırlanmalıdır:
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
Not edin ki, eğer kaydırma penceresi 2 olsaydı, bu, girdi dizisindeki bir sonraki girişin 2 token sonra başlayacağı anlamına gelir ve sadece bir değil, ancak hedef dizi yine de yalnızca 1 token tahmin edecektir. Pytorch'ta, bu kaydırma penceresi `stride` parametresi ile ifade edilir.
