# macOS - AMFI - AppleMobileFileIntegrity

{% hint style="success" %}
Learn & practice AWS Hacking:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Learn & practice GCP Hacking: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Support HackTricks</summary>

* Check the [**subscription plans**](https://github.com/sponsors/carlospolop)!
* **Join the** 💬 [**Discord group**](https://discord.gg/hRep4RUj7f) or the [**telegram group**](https://t.me/peass) or **follow** us on **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Share hacking tricks by submitting PRs to the** [**HackTricks**](https://github.com/carlospolop/hacktricks) and [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repos.

</details>
{% endhint %}



## AppleMobileFileIntegrity.kext ve amfid

Sistemde çalışan kodun bütünlüğünü sağlamak için XNU'nun kod imzası doğrulama mantığını sağlar. Ayrıca, yetkilendirmeleri kontrol edebilir ve hata ayıklama veya görev portlarını elde etme gibi diğer hassas görevleri yönetebilir.

Ayrıca, bazı işlemler için kext, kullanıcı alanında çalışan daemon `/usr/libexec/amfid` ile iletişim kurmayı tercih eder. Bu güven ilişkisi, birkaç jailbreak'te kötüye kullanılmıştır.

AMFI, **MACF** politikalarını kullanır ve başladığı anda kancalarını kaydeder. Ayrıca, yüklenmesini veya boşaltılmasını engellemek bir çekirdek panikine neden olabilir. Ancak, AMFI'yi zayıflatmaya izin veren bazı önyükleme argümanları vardır:

* `amfi_unrestricted_task_for_pid`: Gerekli yetkilendirmeler olmadan task\_for\_pid'ye izin ver
* `amfi_allow_any_signature`: Herhangi bir kod imzasına izin ver
* `cs_enforcement_disable`: Kod imzası uygulamasını devre dışı bırakmak için sistem genelinde kullanılan argüman
* `amfi_prevent_old_entitled_platform_binaries`: Yetkilendirmeleri olan platform ikili dosyalarını geçersiz kıl
* `amfi_get_out_of_my_way`: amfi'yi tamamen devre dışı bırakır

Kaydettiği bazı MACF politikaları şunlardır:

* **`cred_check_label_update_execve:`** Etiket güncellemesi yapılacak ve 1 döndürülecek
* **`cred_label_associate`**: AMFI'nin mac etiket slotunu etiket ile güncelle
* **`cred_label_destroy`**: AMFI’nin mac etiket slotunu kaldır
* **`cred_label_init`**: AMFI'nin mac etiket slotuna 0 yerleştir
* **`cred_label_update_execve`:** Sürecin etiketleri değiştirmesine izin verilip verilmeyeceğini kontrol eder.
* **`file_check_mmap`:** mmap'in bellek alıp almadığını ve bunu çalıştırılabilir olarak ayarlayıp ayarlamadığını kontrol eder. Bu durumda, kütüphane doğrulamasının gerekli olup olmadığını kontrol eder ve gerekiyorsa kütüphane doğrulama fonksiyonunu çağırır.
* **`file_check_library_validation`**: Kütüphane doğrulama fonksiyonunu çağırır, bu fonksiyon diğer şeylerin yanı sıra bir platform ikili dosyasının başka bir platform ikili dosyasını yükleyip yüklemediğini veya sürecin ve yeni yüklenen dosyanın aynı TeamID'ye sahip olup olmadığını kontrol eder. Belirli yetkilendirmeler, herhangi bir kütüphaneyi yüklemeye de izin verecektir.
* **`policy_initbsd`**: Güvenilir NVRAM Anahtarlarını ayarlar
* **`policy_syscall`**: İkili dosyanın sınırsız segmentlere sahip olup olmadığını, çevre değişkenlerine izin verip vermeyeceğini kontrol eder... bu, bir süreç `amfi_check_dyld_policy_self()` ile başlatıldığında da çağrılır.
* **`proc_check_inherit_ipc_ports`**: Bir süreç yeni bir ikili dosya çalıştırdığında, diğer süreçlerin sürecin görev portu üzerinde SEND haklarına sahip olup olmadığını kontrol eder. Platform ikili dosyalarına izin verilir, `get-task-allow` yetkilendirmesi buna izin verir, `task_for_pid-allow` yetkilendirmeleri izinlidir ve aynı TeamID'ye sahip ikili dosyalar.
* **`proc_check_expose_task`**: yetkilendirmeleri zorlar
* **`amfi_exc_action_check_exception_send`**: Bir istisna mesajı hata ayıklayıcıya gönderilir
* **`amfi_exc_action_label_associate & amfi_exc_action_label_copy/populate & amfi_exc_action_label_destroy & amfi_exc_action_label_init & amfi_exc_action_label_update`**: İstisna işleme sırasında etiket yaşam döngüsü (hata ayıklama)
* **`proc_check_get_task`**: `get-task-allow` gibi yetkilendirmeleri kontrol eder, bu diğer süreçlerin görev portlarını almasına izin verir ve `task_for_pid-allow`, bu süreçlerin diğer süreçlerin görev portlarını almasına izin verir. Hiçbiri yoksa, `amfid permitunrestricteddebugging`'i çağırarak izin verilip verilmediğini kontrol eder.
* **`proc_check_mprotect`**: `mprotect` çağrıldığında `VM_PROT_TRUSTED` bayrağı ile reddeder, bu bayrak bölgenin geçerli bir kod imzası varmış gibi muamele edilmesi gerektiğini belirtir.
* **`vnode_check_exec`**: Çalıştırılabilir dosyalar belleğe yüklendiğinde çağrılır ve `cs_hard | cs_kill` ayarlarını yapar, bu da herhangi bir sayfa geçersiz hale gelirse süreci öldürür.
* **`vnode_check_getextattr`**: MacOS: `com.apple.root.installed` ve `isVnodeQuarantined()` kontrol eder
* **`vnode_check_setextattr`**: get + com.apple.private.allow-bless ve iç kurulumcu eşdeğeri yetkilendirmesi
* &#x20;**`vnode_check_signature`**: Yetkilendirmeleri, güvenilir önbelleği ve `amfid` kullanarak kod imzasını kontrol etmek için XNU'yu çağıran kod
* &#x20;**`proc_check_run_cs_invalid`**: `ptrace()` çağrılarını (`PT_ATTACH` ve `PT_TRACE_ME`) keser. `get-task-allow`, `run-invalid-allow` ve `run-unsigned-code` gibi herhangi bir yetkilendirmeyi kontrol eder ve hiçbiri yoksa, hata ayıklamanın izinli olup olmadığını kontrol eder.
* **`proc_check_map_anon`**: mmap **`MAP_JIT`** bayrağı ile çağrıldığında, AMFI `dynamic-codesigning` yetkilendirmesini kontrol eder.

`AMFI.kext` ayrıca diğer çekirdek uzantıları için bir API sunar ve bağımlılıklarını bulmak mümkündür:
```bash
kextstat | grep " 19 " | cut -c2-5,50- | cut -d '(' -f1
Executing: /usr/bin/kmutil showloaded
No variant specified, falling back to release
8   com.apple.kec.corecrypto
19   com.apple.driver.AppleMobileFileIntegrity
22   com.apple.security.sandbox
24   com.apple.AppleSystemPolicy
67   com.apple.iokit.IOUSBHostFamily
70   com.apple.driver.AppleUSBTDM
71   com.apple.driver.AppleSEPKeyStore
74   com.apple.iokit.EndpointSecurity
81   com.apple.iokit.IOUserEthernet
101   com.apple.iokit.IO80211Family
102   com.apple.driver.AppleBCMWLANCore
118   com.apple.driver.AppleEmbeddedUSBHost
134   com.apple.iokit.IOGPUFamily
135   com.apple.AGXG13X
137   com.apple.iokit.IOMobileGraphicsFamily
138   com.apple.iokit.IOMobileGraphicsFamily-DCP
162   com.apple.iokit.IONVMeFamily
```
## amfid

Bu, `AMFI.kext`'in kullanıcı modunda kod imzalarını kontrol etmek için kullanacağı kullanıcı modu çalışan daemon'dur.\
`AMFI.kext`'in daemon ile iletişim kurması için `HOST_AMFID_PORT` üzerinden mach mesajları kullanır; bu özel port `18`'dir.

macOS'ta, root süreçlerin özel portları ele geçirmesi artık mümkün değildir çünkü bunlar `SIP` tarafından korunmaktadır ve yalnızca launchd bunlara erişebilir. iOS'ta, yanıtı geri gönderen sürecin `amfid`'nin CDHash'inin hardcoded olduğu kontrol edilir.

`amfid`'in bir ikiliyi kontrol etmesi istendiğinde ve yanıtı alındığında, bunu hata ayıklayarak ve `mach_msg`'de bir kesme noktası ayarlayarak görebilirsiniz.

Özel port üzerinden bir mesaj alındığında **MIG**, her işlevi çağırdığı işlevine göndermek için kullanılır. Ana işlevler tersine mühendislik ile çözüldü ve kitapta açıklandı.

## Provisioning Profiles

Bir provisioning profili kod imzalamak için kullanılabilir. Kod imzalamak ve test etmek için kullanılabilecek **Geliştirici** profilleri ve tüm cihazlarda kullanılabilecek **Kurumsal** profilleri vardır.

Bir Uygulama Apple Store'a gönderildiğinde, onaylanırsa, Apple tarafından imzalanır ve provisioning profiline artık ihtiyaç duyulmaz.

Bir profil genellikle `.mobileprovision` veya `.provisionprofile` uzantısını kullanır ve şu şekilde dökülebilir:
```bash
openssl asn1parse -inform der -in /path/to/profile

# Or

security cms -D -i /path/to/profile
```
Although sometimes referred as certificated, these provisioning profiles have more than a certificate:

* **AppIDName:** Uygulama Tanımlayıcısı
* **AppleInternalProfile**: Bunu Apple İç Profil olarak belirler
* **ApplicationIdentifierPrefix**: AppIDName'e eklenir (TeamIdentifier ile aynı)
* **CreationDate**: `YYYY-MM-DDTHH:mm:ssZ` formatında tarih
* **DeveloperCertificates**: Base64 verisi olarak kodlanmış (genellikle bir) sertifika dizisi
* **Entitlements**: Bu profil için izin verilen haklar
* **ExpirationDate**: `YYYY-MM-DDTHH:mm:ssZ` formatında son kullanma tarihi
* **Name**: Uygulama Adı, AppIDName ile aynı
* **ProvisionedDevices**: Bu profilin geçerli olduğu UDID'lerin dizisi (geliştirici sertifikaları için)
* **ProvisionsAllDevices**: Bir boolean (kurumsal sertifikalar için true)
* **TeamIdentifier**: Uygulamalar arası etkileşim amaçları için geliştiriciyi tanımlamak için kullanılan (genellikle bir) alfanümerik dize dizisi
* **TeamName**: Geliştiriciyi tanımlamak için kullanılan insan tarafından okunabilir ad
* **TimeToLive**: Sertifikanın geçerliliği (gün cinsinden)
* **UUID**: Bu profil için Evrensel Benzersiz Tanımlayıcı
* **Version**: Şu anda 1 olarak ayarlanmış

Not edin ki, haklar girişi kısıtlı bir haklar seti içerecek ve provisioning profili yalnızca bu belirli hakları verebilecektir, böylece Apple özel haklarını vermekten kaçınılacaktır.

Profillerin genellikle `/var/MobileDeviceProvisioningProfiles` içinde bulunduğunu ve bunları **`security cms -D -i /path/to/profile`** ile kontrol etmenin mümkün olduğunu unutmayın.

## **libmis.dyld**

Bu, `amfid`'in bir şeyin izin verilip verilmeyeceğini sormak için çağırdığı dış kütüphanedir. Bu, geçmişte her şeyi izin veren arka kapılı bir versiyonunu çalıştırarak jailbreak'te kötüye kullanılmıştır.

macOS'ta bu `MobileDevice.framework` içinde bulunmaktadır.

## AMFI Trust Caches

iOS AMFI, ad-hoc imzalanmış bilinen hash'lerin bir listesini, **Trust Cache** olarak adlandırılan ve kext'in `__TEXT.__const` bölümünde bulunan bir listeyi sürdürmektedir. Çok özel ve hassas işlemlerde, bu Trust Cache'i bir dış dosya ile genişletmek mümkündür.

## References

* [**\*OS Internals Volume III**](https://newosxbook.com/home.html)

{% hint style="success" %}
Learn & practice AWS Hacking:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Learn & practice GCP Hacking: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Support HackTricks</summary>

* Check the [**subscription plans**](https://github.com/sponsors/carlospolop)!
* **Join the** 💬 [**Discord group**](https://discord.gg/hRep4RUj7f) or the [**telegram group**](https://t.me/peass) or **follow** us on **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Share hacking tricks by submitting PRs to the** [**HackTricks**](https://github.com/carlospolop/hacktricks) and [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repos.

</details>
{% endhint %}
