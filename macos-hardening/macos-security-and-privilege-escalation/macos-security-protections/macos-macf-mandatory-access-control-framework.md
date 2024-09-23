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

## Basic Information

**MACF** का मतलब है **अनिवार्य पहुँच नियंत्रण ढांचा**, जो एक सुरक्षा प्रणाली है जो ऑपरेटिंग सिस्टम में निर्मित है ताकि आपके कंप्यूटर की सुरक्षा में मदद मिल सके। यह **कठोर नियम निर्धारित करके काम करता है कि कौन या क्या सिस्टम के कुछ हिस्सों, जैसे फ़ाइलें, अनुप्रयोग और सिस्टम संसाधन, तक पहुँच सकता है।** इन नियमों को स्वचालित रूप से लागू करके, MACF सुनिश्चित करता है कि केवल अधिकृत उपयोगकर्ता और प्रक्रियाएँ विशिष्ट क्रियाएँ कर सकें, जिससे अनधिकृत पहुँच या दुर्भावनापूर्ण गतिविधियों का जोखिम कम हो जाता है।

ध्यान दें कि MACF वास्तव में कोई निर्णय नहीं लेता क्योंकि यह केवल **क्रियाओं को अवरुद्ध** करता है, यह निर्णय **नीति मॉड्यूल** (कर्नेल एक्सटेंशन) पर छोड़ देता है जिसे यह कॉल करता है जैसे `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` और `mcxalr.kext`।

### Flow

1. प्रक्रिया एक syscall/mach ट्रैप करती है
2. संबंधित फ़ंक्शन कर्नेल के अंदर कॉल किया जाता है
3. फ़ंक्शन MACF को कॉल करता है
4. MACF उन नीति मॉड्यूलों की जाँच करता है जिन्होंने अपनी नीति में उस फ़ंक्शन को हुक करने के लिए अनुरोध किया
5. MACF संबंधित नीतियों को कॉल करता है
6. नीतियाँ संकेत करती हैं कि वे क्रिया की अनुमति देती हैं या अस्वीकार करती हैं

{% hint style="danger" %}
Apple एकमात्र है जो MAC Framework KPI का उपयोग कर सकता है।
{% endhint %}

### Labels

MACF **लेबल** का उपयोग करता है जिन्हें फिर नीतियों द्वारा जाँच की जाएगी कि क्या उन्हें कुछ पहुँच प्रदान करनी चाहिए या नहीं। लेबल संरचना की घोषणा का कोड [यहाँ](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h) पाया जा सकता है, जिसे फिर **`struct ucred`** के अंदर [**यहाँ**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) **`cr_label`** भाग में उपयोग किया जाता है। लेबल में फ्लैग और **MACF नीतियों द्वारा पॉइंटर्स आवंटित करने के लिए उपयोग किए जा सकने वाले स्लॉट** की संख्या होती है। उदाहरण के लिए, Sandbox कंटेनर प्रोफ़ाइल की ओर इशारा करेगा।

## MACF Policies

एक MACF नीति **कुछ कर्नेल संचालन में लागू करने के लिए नियम और शर्तें परिभाषित करती है।** 

एक कर्नेल एक्सटेंशन एक `mac_policy_conf` संरचना को कॉन्फ़िगर कर सकता है और फिर इसे `mac_policy_register` को कॉल करके पंजीकृत कर सकता है। [यहाँ से](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
यह नीतियों को कॉन्फ़िगर करने वाले कर्नेल एक्सटेंशन की पहचान करना आसान है, `mac_policy_register` कॉल की जांच करके। इसके अलावा, एक्सटेंशन के डिस्सेम्बल की जांच करने पर `mac_policy_conf` स्ट्रक्चर भी पाया जा सकता है।

ध्यान दें कि MACF नीतियों को **डायनामिकली** भी पंजीकृत और अपंजीकृत किया जा सकता है।

`mac_policy_conf` के मुख्य क्षेत्रों में से एक है **`mpc_ops`**। यह क्षेत्र निर्दिष्ट करता है कि नीति किन ऑपरेशनों में रुचि रखती है। ध्यान दें कि इनमें सैकड़ों हैं, इसलिए सभी को शून्य करना संभव है और फिर केवल उन पर ध्यान केंद्रित करना जो नीति में रुचि रखते हैं। [यहां](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac\_policy.h.auto.html) से:
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
Almost all the hooks will be called back by MACF when one of those operations are intercepted. However, **`mpo_policy_*`** hooks are an exception because `mpo_hook_policy_init()` एक कॉलबैक है जो पंजीकरण के समय (तो `mac_policy_register()` के बाद) कॉल किया जाता है और `mpo_hook_policy_initbsd()` को लेट पंजीकरण के दौरान कॉल किया जाता है जब BSD उपप्रणाली सही तरीके से प्रारंभ हो चुकी होती है।

Moreover, the **`mpo_policy_syscall`** hook can be registered by any kext to expose a private **ioctl** style call **interface**. Then, a user client will be able to call `mac_syscall` (#381) specifying as parameters the **policy name** with an integer **code** and optional **arguments**.\
For example, the **`Sandbox.kext`** uses this a lot.

Checking the kext's **`__DATA.__const*`** is possible to identify the `mac_policy_ops` structure used when registering the policy. It's possible to find it because its pointer is at an offset inside `mpo_policy_conf` and also because the amount of NULL pointers that will be in that area.

Moreover, it's also possible to get the list of kexts that have configured a policy by dumping from memory the struct **`_mac_policy_list`** which is updated with every policy that is registered.

## MACF Initialization

MACF को बहुत जल्दी प्रारंभ किया जाता है। इसे XNU के `bootstrap_thread` में सेट किया जाता है: `ipc_bootstrap` के बाद `mac_policy_init()` को कॉल किया जाता है जो `mac_policy_list` को प्रारंभ करता है और कुछ ही क्षणों बाद `mac_policy_initmach()` को कॉल किया जाता है। अन्य चीजों के बीच, यह फ़ंक्शन सभी Apple kexts को प्राप्त करेगा जिनके Info.plist में `AppleSecurityExtension` कुंजी है जैसे `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` और `TMSafetyNet.kext` और उन्हें लोड करता है।

## MACF Callouts

यह MACF के लिए कॉलआउट्स को कोड में ढूंढना सामान्य है जैसे: **`#if CONFIG_MAC`** कंडीशनल ब्लॉक्स। इसके अलावा, इन ब्लॉक्स के अंदर `mac_proc_check*` को कॉल करने के लिए MACF को **अनुमतियों की जांच** करने के लिए कॉल किया जाता है। इसके अलावा, MACF कॉलआउट्स का प्रारूप है: **`mac_<object>_<opType>_opName`**।

The object is one of the following: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
The `opType` is usually check which will be used to allow or deny the action. However, it's also possible to find `notify`, which will allow the kext to react to the given action.

You can find an example in [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621):

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
जो `MAC_CHECK` मैक्रो को कॉल कर रहा है, जिसका कोड [यहां](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac_internal.h#L261) पाया जा सकता है।
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
जो सभी पंजीकृत मैक नीतियों को उनके कार्यों को कॉल करते हुए और आउटपुट को त्रुटि चर में संग्रहीत करते हुए जाएगा, जिसे केवल `mac_error_select` द्वारा सफलता कोड के द्वारा ओवरराइड किया जा सकेगा, इसलिए यदि कोई जांच विफल होती है तो पूरी जांच विफल हो जाएगी और क्रिया की अनुमति नहीं दी जाएगी।

{% hint style="success" %}
हालांकि, याद रखें कि सभी MACF कॉलआउट केवल क्रियाओं को अस्वीकार करने के लिए उपयोग नहीं किए जाते हैं। उदाहरण के लिए, `mac_priv_grant` मैक्रो [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L274) को कॉल करता है, जो यदि कोई नीति 0 के साथ उत्तर देती है तो अनुरोधित विशेषाधिकार को प्रदान करेगा:
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

ये कॉल्स उन (दर्जनों) **privileges** की जांच और प्रदान करने के लिए हैं जो [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h) में परिभाषित हैं।\
कुछ कर्नेल कोड `priv_check_cred()` को [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) से प्रक्रिया के KAuth क्रेडेंशियल्स के साथ और एक प्रिविलेज कोड के साथ कॉल करेगा जो `mac_priv_check` को कॉल करेगा यह देखने के लिए कि क्या कोई नीति **denies** प्रिविलेज देने से रोकती है और फिर यह `mac_priv_grant` को कॉल करेगा यह देखने के लिए कि क्या कोई नीति `privilege` को प्रदान करती है।

### proc\_check\_syscall\_unix

यह हुक सभी सिस्टम कॉल्स को इंटरसेप्ट करने की अनुमति देता है। `bsd/dev/[i386|arm]/systemcalls.c` में घोषित फ़ंक्शन [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25) को देखना संभव है, जिसमें यह कोड है:
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
जो कॉलिंग प्रक्रिया **bitmask** में जांच करेगा कि क्या वर्तमान syscall को `mac_proc_check_syscall_unix` को कॉल करना चाहिए। इसका कारण यह है कि syscalls इतनी बार कॉल किए जाते हैं कि हर बार `mac_proc_check_syscall_unix` को कॉल करने से बचना दिलचस्प है।

ध्यान दें कि फ़ंक्शन `proc_set_syscall_filter_mask()` जो एक प्रक्रिया में bitmask syscalls सेट करता है, Sandbox द्वारा सैंडबॉक्स किए गए प्रक्रियाओं पर मास्क सेट करने के लिए कॉल किया जाता है।

## एक्सपोज़ MACF syscalls

MACF के साथ इंटरैक्ट करना संभव है कुछ syscalls के माध्यम से जो [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151) में परिभाषित हैं:
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
## संदर्भ

* [**\*OS आंतरिक भाग वॉल्यूम III**](https://newosxbook.com/home.html)

{% hint style="success" %}
सीखें और AWS हैकिंग का अभ्यास करें:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks प्रशिक्षण AWS रेड टीम विशेषज्ञ (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
सीखें और GCP हैकिंग का अभ्यास करें: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks प्रशिक्षण GCP रेड टीम विशेषज्ञ (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>HackTricks का समर्थन करें</summary>

* [**सदस्यता योजनाएँ**](https://github.com/sponsors/carlospolop) देखें!
* **हमारे** 💬 [**Discord समूह**](https://discord.gg/hRep4RUj7f) या [**टेलीग्राम समूह**](https://t.me/peass) में शामिल हों या **हमें** **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)** पर फॉलो करें।**
* **हैकिंग ट्रिक्स साझा करें और** [**HackTricks**](https://github.com/carlospolop/hacktricks) और [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) गिटहब रिपोजिटरी में PR सबमिट करें।

</details>
{% endhint %}
