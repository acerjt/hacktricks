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

## Informations de base

**MACF** signifie **Mandatory Access Control Framework**, qui est un système de sécurité intégré au système d'exploitation pour aider à protéger votre ordinateur. Il fonctionne en établissant **des règles strictes sur qui ou quoi peut accéder à certaines parties du système**, telles que des fichiers, des applications et des ressources système. En appliquant automatiquement ces règles, MACF garantit que seuls les utilisateurs et processus autorisés peuvent effectuer des actions spécifiques, réduisant ainsi le risque d'accès non autorisé ou d'activités malveillantes.

Notez que MACF ne prend pas vraiment de décisions, car il **intercepte** simplement les actions, laissant les décisions aux **modules de politique** (extensions du noyau) qu'il appelle comme `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` et `mcxalr.kext`.

### Flux

1. Le processus effectue un appel système/trap mach
2. La fonction pertinente est appelée à l'intérieur du noyau
3. La fonction appelle MACF
4. MACF vérifie les modules de politique qui ont demandé à accrocher cette fonction dans leur politique
5. MACF appelle les politiques pertinentes
6. Les politiques indiquent si elles autorisent ou refusent l'action

{% hint style="danger" %}
Apple est le seul à pouvoir utiliser le KPI du cadre MAC.
{% endhint %}

### Étiquettes

MACF utilise des **étiquettes** que les politiques utiliseront ensuite pour vérifier si elles doivent accorder un accès ou non. Le code de la déclaration de structure des étiquettes peut être [trouvé ici](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h), qui est ensuite utilisé à l'intérieur de la **`struct ucred`** [**ici**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) dans la partie **`cr_label`**. L'étiquette contient des indicateurs et un nombre de **slots** qui peuvent être utilisés par **les politiques MACF pour allouer des pointeurs**. Par exemple, Sanbox pointera vers le profil du conteneur.

## Politiques MACF

Une politique MACF définit **des règles et des conditions à appliquer dans certaines opérations du noyau**.&#x20;

Une extension de noyau pourrait configurer une structure `mac_policy_conf` puis l'enregistrer en appelant `mac_policy_register`. À partir de [ici](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
Il est facile d'identifier les extensions du noyau configurant ces politiques en vérifiant les appels à `mac_policy_register`. De plus, en vérifiant le désassemblage de l'extension, il est également possible de trouver la structure `mac_policy_conf` utilisée.

Notez que les politiques MACF peuvent également être enregistrées et désenregistrées **dynamiquement**.

L'un des principaux champs de la `mac_policy_conf` est le **`mpc_ops`**. Ce champ spécifie les opérations qui intéressent la politique. Notez qu'il y en a des centaines, il est donc possible de les mettre toutes à zéro et ensuite de sélectionner uniquement celles qui intéressent la politique. D'après [ici](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac\_policy.h.auto.html) :
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
Presque tous les hooks seront rappelés par MACF lorsque l'une de ces opérations est interceptée. Cependant, les hooks **`mpo_policy_*`** sont une exception car `mpo_hook_policy_init()` est un rappel appelé lors de l'enregistrement (donc après `mac_policy_register()`) et `mpo_hook_policy_initbsd()` est appelé lors de l'enregistrement tardif une fois que le sous-système BSD a été correctement initialisé.

De plus, le hook **`mpo_policy_syscall`** peut être enregistré par n'importe quel kext pour exposer un appel de style **ioctl** **interface** privé. Ensuite, un client utilisateur pourra appeler `mac_syscall` (#381) en spécifiant comme paramètres le **nom de la politique** avec un **code** entier et des **arguments** optionnels.\
Par exemple, le **`Sandbox.kext`** utilise cela beaucoup.

Vérifier le **`__DATA.__const*`** du kext est possible pour identifier la structure `mac_policy_ops` utilisée lors de l'enregistrement de la politique. Il est possible de la trouver car son pointeur est à un décalage à l'intérieur de `mpo_policy_conf` et aussi à cause du nombre de pointeurs NULL qui seront dans cette zone.

De plus, il est également possible d'obtenir la liste des kexts qui ont configuré une politique en vidant de la mémoire la structure **`_mac_policy_list`** qui est mise à jour avec chaque politique qui est enregistrée.

## Initialisation de MACF

MACF est initialisé très tôt. Il est configuré dans le `bootstrap_thread` de XNU : après `ipc_bootstrap`, un appel à `mac_policy_init()` qui initialise la `mac_policy_list` et quelques instants plus tard `mac_policy_initmach()` est appelé. Parmi d'autres choses, cette fonction obtiendra tous les kexts Apple avec la clé `AppleSecurityExtension` dans leur Info.plist comme `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` et `TMSafetyNet.kext` et les charge.

## Appels MACF

Il est courant de trouver des appels à MACF définis dans le code comme : **`#if CONFIG_MAC`** blocs conditionnels. De plus, à l'intérieur de ces blocs, il est possible de trouver des appels à `mac_proc_check*` qui appelle MACF pour **vérifier les permissions** pour effectuer certaines actions. De plus, le format des appels MACF est : **`mac_<object>_<opType>_opName`**.

L'objet est l'un des suivants : `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
Le `opType` est généralement check qui sera utilisé pour autoriser ou refuser l'action. Cependant, il est également possible de trouver `notify`, ce qui permettra au kext de réagir à l'action donnée.

Vous pouvez trouver un exemple dans [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621):

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

Ensuite, il est possible de trouver le code de `mac_file_check_mmap` dans [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174)
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
Qui appelle le macro `MAC_CHECK`, dont le code peut être trouvé dans [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261)
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
Qui passera en revue toutes les politiques mac enregistrées en appelant leurs fonctions et en stockant la sortie dans la variable d'erreur, qui ne pourra être remplacée que par `mac_error_select` par des codes de succès, donc si un contrôle échoue, le contrôle complet échouera et l'action ne sera pas autorisée.

{% hint style="success" %}
Cependant, rappelez-vous que tous les appels MACF ne sont pas utilisés uniquement pour refuser des actions. Par exemple, `mac_priv_grant` appelle le macro [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L274), qui accordera le privilège demandé si une politique répond avec un 0 :
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

Ces appels sont destinés à vérifier et à fournir (des dizaines de) **privileges** définis dans [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h).\
Certaines parties du code du noyau appelleraient `priv_check_cred()` depuis [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) avec les informations d'identification KAuth du processus et l'un des codes de privilège, ce qui appellera `mac_priv_check` pour voir si une politique **refuse** de donner le privilège, puis il appelle `mac_priv_grant` pour voir si une politique accorde le `privilege`.

### proc\_check\_syscall\_unix

Ce hook permet d'intercepter tous les appels système. Dans `bsd/dev/[i386|arm]/systemcalls.c`, il est possible de voir la fonction déclarée [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25), qui contient ce code :
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
Qui vérifiera dans le **bitmask** du processus appelant si l'appel système actuel doit appeler `mac_proc_check_syscall_unix`. Cela est dû au fait que les appels système sont effectués si fréquemment qu'il est intéressant d'éviter d'appeler `mac_proc_check_syscall_unix` à chaque fois.

Notez que la fonction `proc_set_syscall_filter_mask()`, qui définit le bitmask des appels système dans un processus, est appelée par Sandbox pour définir des masques sur les processus isolés.

## Appels système MACF exposés

Il est possible d'interagir avec MACF via certains appels système définis dans [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151):
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
## Références

* [**\*OS Internals Volume III**](https://newosxbook.com/home.html)

{% hint style="success" %}
Apprenez et pratiquez le hacking AWS :<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Apprenez et pratiquez le hacking GCP : <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Soutenir HackTricks</summary>

* Consultez les [**plans d'abonnement**](https://github.com/sponsors/carlospolop) !
* **Rejoignez le** 💬 [**groupe Discord**](https://discord.gg/hRep4RUj7f) ou le [**groupe telegram**](https://t.me/peass) ou **suivez-nous sur** **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Partagez des astuces de hacking en soumettant des PRs aux** [**HackTricks**](https://github.com/carlospolop/hacktricks) et [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) dépôts github.

</details>
{% endhint %}
