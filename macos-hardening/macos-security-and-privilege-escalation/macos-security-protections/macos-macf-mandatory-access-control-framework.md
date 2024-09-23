# macOS MACF

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

## Temel Bilgiler

**MACF**, **Zorunlu Erişim Kontrol Çerçevesi** anlamına gelir ve bilgisayarınızı korumaya yardımcı olmak için işletim sistemine entegre edilmiş bir güvenlik sistemidir. Belirli sistem bölümlerine, dosyalara, uygulamalara ve sistem kaynaklarına kimlerin veya nelerin erişebileceği konusunda **katı kurallar belirleyerek** çalışır. Bu kuralları otomatik olarak uygulayarak, MACF yalnızca yetkilendirilmiş kullanıcıların ve süreçlerin belirli eylemleri gerçekleştirmesine izin verir, yetkisiz erişim veya kötü niyetli faaliyetler riskini azaltır.

MACF'nin gerçekten herhangi bir karar vermediğini, yalnızca eylemleri **yakaladığını** ve kararları çağırdığı **politika modüllerine** (kernel uzantıları) bıraktığını unutmayın; bunlar arasında `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` ve `mcxalr.kext` bulunmaktadır.

### Akış

1. Süreç bir syscall/mach tuzağı gerçekleştirir
2. İlgili işlev çekirdek içinde çağrılır
3. İşlev MACF'yi çağırır
4. MACF, o işlevi politikalarına bağlamak için talep eden politika modüllerini kontrol eder
5. MACF, ilgili politikaları çağırır
6. Politikalar, eylemi izin verip vermeyeceklerini belirtir

{% hint style="danger" %}
Apple, MAC Framework KPI'sini kullanabilen tek kişidir.
{% endhint %}

### Etiketler

MACF, ardından politikaların bazı erişim izni verip vermeyeceğini kontrol edeceği **etiketler** kullanır. Etiketlerin yapı tanımının kodu [burada](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h) bulunabilir; bu, **`struct ucred`** içinde [**burada**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) **`cr_label`** kısmında kullanılır. Etiket, **MACF politikalarının işaretçi ayırması için** kullanabileceği bayraklar ve bir dizi **slot** içerir. Örneğin, Sandbox konteyner profilini işaret edecektir.

## MACF Politikaları

Bir MACF Politikası, belirli çekirdek işlemlerinde uygulanacak **kural ve koşulları** tanımlar.&#x20;

Bir çekirdek uzantısı, bir `mac_policy_conf` yapısını yapılandırabilir ve ardından `mac_policy_register` çağrısını yaparak kaydedebilir. [Buradan](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
```c
#define mpc_t	struct mac_policy_conf *

/**
@brief Mac policy configuration

This structure specifies the configuration information for a
MAC policy module.  A policy module developer must supply
a short unique policy name, a more descriptive full name, a list of label
namespaces and count, a pointer to the registered enty point operations,
any load time flags, and optionally, a pointer to a label slot identifier.

The Framework will update the runtime flags (mpc_runtime_flags) to
indicate that the module has been registered.

If the label slot identifier (mpc_field_off) is NULL, the Framework
will not provide label storage for the policy.  Otherwise, the
Framework will store the label location (slot) in this field.

The mpc_list field is used by the Framework and should not be
modified by policies.
*/
/* XXX - reorder these for better aligment on 64bit platforms */
struct mac_policy_conf {
const char		*mpc_name;		/** policy name */
const char		*mpc_fullname;		/** full name */
const char		**mpc_labelnames;	/** managed label namespaces */
unsigned int		 mpc_labelname_count;	/** number of managed label namespaces */
struct mac_policy_ops	*mpc_ops;		/** operation vector */
int			 mpc_loadtime_flags;	/** load time flags */
int			*mpc_field_off;		/** label slot */
int			 mpc_runtime_flags;	/** run time flags */
mpc_t			 mpc_list;		/** List reference */
void			*mpc_data;		/** module data */
};
```
Kernel uzantılarını bu politikaları yapılandırırken tanımlamak, `mac_policy_register` çağrılarına bakarak kolaydır. Ayrıca, uzantının ayrıştırmasını kontrol ederek kullanılan `mac_policy_conf` yapısını bulmak da mümkündür.

MACF politikalarının **dinamik** olarak kaydedilebileceğini ve kaydının kaldırılabileceğini unutmayın.

`mac_policy_conf`'nin ana alanlarından biri **`mpc_ops`**'dir. Bu alan, politikanın ilgilendiği işlemleri belirtir. Bunların yüzlercesi olduğunu unutmayın, bu nedenle hepsini sıfırlamak ve ardından politikanın ilgilendiği yalnızca belirli olanları seçmek mümkündür. [Buradan](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
```c
struct mac_policy_ops {
mpo_audit_check_postselect_t		*mpo_audit_check_postselect;
mpo_audit_check_preselect_t		*mpo_audit_check_preselect;
mpo_bpfdesc_label_associate_t		*mpo_bpfdesc_label_associate;
mpo_bpfdesc_label_destroy_t		*mpo_bpfdesc_label_destroy;
mpo_bpfdesc_label_init_t		*mpo_bpfdesc_label_init;
mpo_bpfdesc_check_receive_t		*mpo_bpfdesc_check_receive;
mpo_cred_check_label_update_execve_t	*mpo_cred_check_label_update_execve;
mpo_cred_check_label_update_t		*mpo_cred_check_label_update;
[...]
```
Almost all the hooks will be called back by MACF when one of those operations are intercepted. However, **`mpo_policy_*`** hooks are an exception because `mpo_hook_policy_init()` is a callback called upon registration (so after `mac_policy_register()`) and `mpo_hook_policy_initbsd()` is called during late registration once the BSD subsystem has initialised properly.

Moreover, the **`mpo_policy_syscall`** hook can be registered by any kext to expose a private **ioctl** style call **interface**. Then, a user client will be able to call `mac_syscall` (#381) specifying as parameters the **policy name** with an integer **code** and optional **arguments**.\
For example, the **`Sandbox.kext`** uses this a lot.

Checking the kext's **`__DATA.__const*`** is possible to identify the `mac_policy_ops` structure used when registering the policy. It's possible to find it because its pointer is at an offset inside `mpo_policy_conf` and also because the amount of NULL pointers that will be in that area.

Moreover, it's also possible to get the list of kexts that have configured a policy by dumping from memory the struct **`_mac_policy_list`** which is updated with every policy that is registered.

## MACF Initialization

MACF çok kısa bir süre içinde başlatılır. XNU'nun `bootstrap_thread`'inde ayarlanır: `ipc_bootstrap`'tan sonra `mac_policy_init()` çağrısı yapılır, bu da `mac_policy_list`'i başlatır ve kısa bir süre sonra `mac_policy_initmach()` çağrılır. Bu fonksiyon, `Info.plist`'lerinde `AppleSecurityExtension` anahtarına sahip tüm Apple kext'lerini alır, örneğin `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` ve `TMSafetyNet.kext` ve bunları yükler.

## MACF Callouts

MACF'ye yapılan çağrıları **`#if CONFIG_MAC`** gibi kod içinde tanımlanmış bloklarda bulmak yaygındır. Ayrıca, bu blokların içinde belirli eylemleri gerçekleştirmek için izinleri **kontrol etmek** amacıyla MACF'yi çağıran `mac_proc_check*` çağrılarını bulmak mümkündür. Ayrıca, MACF çağrılarının formatı: **`mac_<object>_<opType>_opName`** şeklindedir.

Nesne aşağıdakilerden biri olabilir: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
`opType` genellikle eylemi izin vermek veya reddetmek için kullanılacak olan kontrol anlamına gelir. Ancak, verilen eyleme tepki vermek için kext'in izin vereceği `notify`'yi bulmak da mümkündür.

Bir örneği [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621) adresinde bulabilirsiniz:

<pre class="language-c"><code class="lang-c">int
mmap(proc_t p, struct mmap_args *uap, user_addr_t *retval)
{
[...]
#if CONFIG_MACF
<strong>			error = mac_file_check_mmap(vfs_context_ucred(ctx),
</strong>			    fp->fp_glob, prot, flags, file_pos + pageoff,
&#x26;maxprot);
if (error) {
(void)vnode_put(vp);
goto bad;
}
#endif /* MAC */
[...]
</code></pre>

Then, it's possible to find the code of `mac_file_check_mmap` in [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174)
```c
mac_file_check_mmap(struct ucred *cred, struct fileglob *fg, int prot,
int flags, uint64_t offset, int *maxprot)
{
int error;
int maxp;

maxp = *maxprot;
MAC_CHECK(file_check_mmap, cred, fg, NULL, prot, flags, offset, &maxp);
if ((maxp | *maxprot) != *maxprot) {
panic("file_check_mmap increased max protections");
}
*maxprot = maxp;
return error;
}
```
`MAC_CHECK` makrosunu çağıran, kodu [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261) adresinde bulunabilir.
```c
/*
* MAC_CHECK performs the designated check by walking the policy
* module list and checking with each as to how it feels about the
* request.  Note that it returns its value via 'error' in the scope
* of the caller.
*/
#define MAC_CHECK(check, args...) do {                              \
error = 0;                                                      \
MAC_POLICY_ITERATE({                                            \
if (mpc->mpc_ops->mpo_ ## check != NULL) {              \
DTRACE_MACF3(mac__call__ ## check, void *, mpc, int, error, int, MAC_ITERATE_CHECK); \
int __step_err = mpc->mpc_ops->mpo_ ## check (args); \
DTRACE_MACF2(mac__rslt__ ## check, void *, mpc, int, __step_err); \
error = mac_error_select(__step_err, error);         \
}                                                           \
});                                                             \
} while (0)
```
Hangi, tüm kayıtlı mac politikalarını çağırarak işlevlerini çalıştıracak ve çıktıyı hata değişkeninde depolayacak, bu değişken yalnızca başarı kodları ile `mac_error_select` tarafından geçersiz kılınabilir, bu nedenle herhangi bir kontrol başarısız olursa, tüm kontrol başarısız olacak ve işlem izin verilmeyecektir.

{% hint style="success" %}
Ancak, tüm MACF çağrılarının yalnızca eylemleri reddetmek için kullanılmadığını unutmayın. Örneğin, `mac_priv_grant`, herhangi bir politikanın 0 ile yanıt vermesi durumunda talep edilen ayrıcalığı verecek olan [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac_internal.h#L274) makrosunu çağırır:
```c
/*
* MAC_GRANT performs the designated check by walking the policy
* module list and checking with each as to how it feels about the
* request.  Unlike MAC_CHECK, it grants if any policies return '0',
* and otherwise returns EPERM.  Note that it returns its value via
* 'error' in the scope of the caller.
*/
#define MAC_GRANT(check, args...) do {                              \
error = EPERM;                                                  \
MAC_POLICY_ITERATE({                                            \
if (mpc->mpc_ops->mpo_ ## check != NULL) {                  \
DTRACE_MACF3(mac__call__ ## check, void *, mpc, int, error, int, MAC_ITERATE_GRANT); \
int __step_res = mpc->mpc_ops->mpo_ ## check (args); \
if (__step_res == 0) {                              \
error = 0;                                  \
}                                                   \
DTRACE_MACF2(mac__rslt__ ## check, void *, mpc, int, __step_res); \
}                                                           \
});                                                             \
} while (0)
```
{% endhint %}

### priv\_check & priv\_grant

Bu çağrılar, [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h) dosyasında tanımlanan (onlarca) **ayrıcalığı** kontrol etmek ve sağlamak için tasarlanmıştır.\
Bazı çekirdek kodları, sürecin KAuth kimlik bilgileri ile `priv_check_cred()` çağrısını [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) dosyasından yapar ve ayrıcalık kodlarından birini kullanarak `mac_priv_check` çağrısını yapar; bu, herhangi bir politikanın ayrıcalığı vermeyi **reddedip** etmediğini kontrol eder ve ardından `mac_priv_grant` çağrısını yaparak herhangi bir politikanın `ayrıcalığı` verip vermediğini kontrol eder.

### proc\_check\_syscall\_unix

Bu kanca, tüm sistem çağrılarını yakalamaya olanak tanır. `bsd/dev/[i386|arm]/systemcalls.c` dosyasında, bu kodu içeren tanımlı fonksiyonu [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25) görebilirsiniz:
```c
#if CONFIG_MACF
if (__improbable(proc_syscall_filter_mask(proc) != NULL && !bitstr_test(proc_syscall_filter_mask(proc), syscode))) {
error = mac_proc_check_syscall_unix(proc, syscode);
if (error) {
goto skip_syscall;
}
}
#endif /* CONFIG_MACF */
```
Hangi çağıran süreçte **bitmask** kontrol edilecektir, eğer mevcut syscall `mac_proc_check_syscall_unix` çağırması gerekiyorsa. Bunun nedeni, syscalls'ın bu kadar sık çağrılmasıdır, bu yüzden her seferinde `mac_proc_check_syscall_unix` çağırmaktan kaçınmak ilginçtir.

`proc_set_syscall_filter_mask()` fonksiyonunun, bir süreçte bitmask syscalls'ı ayarlamak için Sandbox tarafından çağrıldığını unutmayın; bu, sandboxed süreçlerde maskeleri ayarlamak içindir.

## Açık MACF syscalls

MACF ile etkileşimde bulunmak, [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151) dosyasında tanımlanan bazı syscalls aracılığıyla mümkündür:
```c
/*
* Extended non-POSIX.1e interfaces that offer additional services
* available from the userland and kernel MAC frameworks.
*/
#ifdef __APPLE_API_PRIVATE
__BEGIN_DECLS
int      __mac_execve(char *fname, char **argv, char **envv, mac_t _label);
int      __mac_get_fd(int _fd, mac_t _label);
int      __mac_get_file(const char *_path, mac_t _label);
int      __mac_get_link(const char *_path, mac_t _label);
int      __mac_get_pid(pid_t _pid, mac_t _label);
int      __mac_get_proc(mac_t _label);
int      __mac_set_fd(int _fildes, const mac_t _label);
int      __mac_set_file(const char *_path, mac_t _label);
int      __mac_set_link(const char *_path, mac_t _label);
int      __mac_mount(const char *type, const char *path, int flags, void *data,
struct mac *label);
int      __mac_get_mount(const char *path, struct mac *label);
int      __mac_set_proc(const mac_t _label);
int      __mac_syscall(const char *_policyname, int _call, void *_arg);
__END_DECLS
#endif /*__APPLE_API_PRIVATE*/
```
## Referanslar

* [**\*OS İç Yapıları Cilt III**](https://newosxbook.com/home.html)

{% hint style="success" %}
AWS Hacking öğrenin ve pratik yapın:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Eğitim AWS Kırmızı Ekip Uzmanı (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
GCP Hacking öğrenin ve pratik yapın: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Eğitim GCP Kırmızı Ekip Uzmanı (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>HackTricks'i Destekleyin</summary>

* [**abonelik planlarını**](https://github.com/sponsors/carlospolop) kontrol edin!
* **💬 [**Discord grubuna**](https://discord.gg/hRep4RUj7f) veya [**telegram grubuna**](https://t.me/peass) katılın ya da **Twitter'da** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**'i takip edin.**
* **Hacking ipuçlarını paylaşmak için** [**HackTricks**](https://github.com/carlospolop/hacktricks) ve [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github reposuna PR gönderin.

</details>
{% endhint %}
