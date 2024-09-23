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

**MACF** inamaanisha **Mandatory Access Control Framework**, ambayo ni mfumo wa usalama uliojengwa ndani ya mfumo wa uendeshaji kusaidia kulinda kompyuta yako. Inafanya kazi kwa kuweka **kanuni kali kuhusu nani au nini kinaweza kufikia sehemu fulani za mfumo**, kama vile faili, programu, na rasilimali za mfumo. Kwa kutekeleza kanuni hizi kiotomatiki, MACF inahakikisha kwamba ni watumiaji na michakato walioidhinishwa pekee wanaweza kufanya vitendo maalum, kupunguza hatari ya ufikiaji usioidhinishwa au shughuli mbaya.

Kumbuka kwamba MACF haifanyi maamuzi yoyote kwani inachukua tu **hatua**, inawaachia maamuzi **moduli za sera** (kernel extensions) inazopiga simu kama `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` na `mcxalr.kext`.

### Flow

1. Mchakato unafanya syscall/mach trap
2. Kazi husika inaitwa ndani ya kernel
3. Kazi inaita MACF
4. MACF inakagua moduli za sera ambazo zilitaka kuunganisha kazi hiyo katika sera zao
5. MACF inaita sera husika
6. Sera zinaonyesha kama zinaruhusu au kukataa hatua hiyo

{% hint style="danger" %}
Apple ndiye pekee anayeweza kutumia MAC Framework KPI.
{% endhint %}

### Labels

MACF inatumia **labels** ambazo kisha sera zinakagua kama zinapaswa kutoa ufikiaji fulani au la. Kanuni ya kutangaza muundo wa labels inaweza kupatikana [hapa](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h), ambayo kisha inatumika ndani ya **`struct ucred`** [**hapa**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) katika sehemu ya **`cr_label`**. Label ina bendera na nambari ya **slots** ambazo zinaweza kutumika na **sera za MACF kutoa viashiria**. Kwa mfano, Sanbox itakuwa na kiashiria cha wasifu wa kontena.

## MACF Policies

Sera ya MACF inafafanua **kanuni na masharti yanayopaswa kutumika katika operesheni fulani za kernel**.&#x20;

Kupanua kernel kunaweza kuunda muundo wa `mac_policy_conf` kisha kujiandikisha kwa kuita `mac_policy_register`. Kutoka [hapa](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
Ni rahisi kubaini nyongeza za kernel zinazokamilisha sera hizi kwa kuangalia simu za `mac_policy_register`. Zaidi ya hayo, kuangalia disassemble ya nyongeza pia inawezekana kupata muundo wa `mac_policy_conf` unaotumika.

Kumbuka kwamba sera za MACF zinaweza kuandikishwa na kufutwa pia **kitaalamu**.

Moja ya maeneo makuu ya `mac_policy_conf` ni **`mpc_ops`**. Huu ni uwanja unaoeleza ni shughuli zipi sera inavutiwa nazo. Kumbuka kwamba kuna mamia yao, hivyo inawezekana kuweka sifuri zote kisha kuchagua zile tu ambazo sera inavutiwa nazo. Kutoka [hapa](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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

MACF inaanzishwa mapema sana. Inapangwa katika `bootstrap_thread` ya XNU: baada ya `ipc_bootstrap` wito wa `mac_policy_init()` ambayo inaanzisha `mac_policy_list` na muda mfupi baadaye `mac_policy_initmach()` inaitwa. Miongoni mwa mambo mengine, kazi hii itapata kexts zote za Apple zenye ufunguo wa `AppleSecurityExtension` katika Info.plist yao kama vile `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` na `TMSafetyNet.kext` na kuzipakia.

## MACF Callouts

Ni kawaida kupata wito kwa MACF ulioainishwa katika msimbo kama: **`#if CONFIG_MAC`** vizuizi vya masharti. Aidha, ndani ya vizuizi hivi inawezekana kupata wito kwa `mac_proc_check*` ambayo inaita MACF ili **kuangalia ruhusa** za kutekeleza vitendo fulani. Aidha, muundo wa wito wa MACF ni: **`mac_<object>_<opType>_opName`**.

Kitu ni kimoja kati ya yafuatayo: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
`opType` kwa kawaida ni check ambayo itatumika kuruhusu au kukataa kitendo. Hata hivyo, pia inawezekana kupata `notify`, ambayo itaruhusu kext kujibu kitendo kilichotolewa.

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
Ambayo inaita macro ya `MAC_CHECK`, ambayo msimbo wake unaweza kupatikana katika [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261)
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
Ambayo itapitia sera zote za mac zilizorekodiwa ikitumia kazi zao na kuhifadhi matokeo ndani ya mabadiliko ya makosa, ambayo yanaweza kubadilishwa tu na `mac_error_select` kwa nambari za mafanikio hivyo ikiwa ukaguzi wowote unashindwa ukaguzi kamili utashindwa na hatua haitaruhusiwa.

{% hint style="success" %}
Hata hivyo, kumbuka kwamba si kila kito cha MACF kinatumika tu kukataa hatua. Kwa mfano, `mac_priv_grant` inaita macro [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac_internal.h#L274), ambayo itatoa kibali kilichohitajika ikiwa sera yoyote itajibu kwa 0:
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

Hizi callas zinakusudia kuangalia na kutoa (mifumo ya) **privileges** zilizofafanuliwa katika [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h).\
Baadhi ya msimbo wa kernel utaita `priv_check_cred()` kutoka [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) kwa kutumia KAuth credentials za mchakato na moja ya msimbo wa privileges ambayo itaita `mac_priv_check` ili kuona kama sera yoyote **inasitisha** kutoa ile privilege na kisha inaita `mac_priv_grant` ili kuona kama sera yoyote inatoa `privilege`.

### proc\_check\_syscall\_unix

Hii hook inaruhusu kukamata simu zote za mfumo. Katika `bsd/dev/[i386|arm]/systemcalls.c` inawezekana kuona kazi iliyoelezwa [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25), ambayo ina msimbo huu:
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
Ambayo itakagua katika mchakato unaoitwa **bitmask** ikiwa syscall ya sasa inapaswa kuita `mac_proc_check_syscall_unix`. Hii ni kwa sababu syscalls zinaitwa mara nyingi sana kwamba ni muhimu kuepuka kuita `mac_proc_check_syscall_unix` kila wakati.

Kumbuka kwamba kazi `proc_set_syscall_filter_mask()`, ambayo huweka bitmask syscalls katika mchakato inaitwa na Sandbox kuweka masks kwenye mchakato zilizowekwa kwenye sandbox.

## Syscalls za MACF zilizofichuliwa

Inawezekana kuingiliana na MACF kupitia baadhi ya syscalls zilizofafanuliwa katika [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151):
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
## References

* [**\*OS Internals Volume III**](https://newosxbook.com/home.html)

{% hint style="success" %}
Jifunze na fanya mazoezi ya AWS Hacking:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Jifunze na fanya mazoezi ya GCP Hacking: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Support HackTricks</summary>

* Angalia [**mpango wa usajili**](https://github.com/sponsors/carlospolop)!
* **Jiunge na** 💬 [**kikundi cha Discord**](https://discord.gg/hRep4RUj7f) au [**kikundi cha telegram**](https://t.me/peass) au **tufuatilie** kwenye **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Shiriki mbinu za hacking kwa kuwasilisha PRs kwa** [**HackTricks**](https://github.com/carlospolop/hacktricks) na [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) repos za github.

</details>
{% endhint %}
