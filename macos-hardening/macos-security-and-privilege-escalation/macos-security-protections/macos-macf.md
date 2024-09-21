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

**MACF** σημαίνει **Mandatory Access Control Framework**, το οποίο είναι ένα σύστημα ασφαλείας ενσωματωμένο στο λειτουργικό σύστημα για να βοηθήσει στην προστασία του υπολογιστή σας. Λειτουργεί θέτοντας **αυστηρούς κανόνες σχετικά με το ποιος ή τι μπορεί να έχει πρόσβαση σε ορισμένα μέρη του συστήματος**, όπως αρχεία, εφαρμογές και πόρους συστήματος. Εφαρμόζοντας αυτούς τους κανόνες αυτόματα, το MACF διασφαλίζει ότι μόνο οι εξουσιοδοτημένοι χρήστες και διαδικασίες μπορούν να εκτελούν συγκεκριμένες ενέργειες, μειώνοντας τον κίνδυνο μη εξουσιοδοτημένης πρόσβασης ή κακόβουλων δραστηριοτήτων.

Σημειώστε ότι το MACF δεν παίρνει πραγματικά αποφάσεις καθώς απλώς **παρεμβαίνει** σε ενέργειες, αφήνει τις αποφάσεις στα **πολιτικά modules** (επέκταση πυρήνα) που καλεί όπως `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` και `mcxalr.kext`.

### Flow

1. Η διαδικασία εκτελεί μια syscall/mach trap
2. Η σχετική λειτουργία καλείται μέσα στον πυρήνα
3. Η λειτουργία καλεί το MACF
4. Το MACF ελέγχει τα πολιτικά modules που ζήτησαν να συνδεθούν με αυτή τη λειτουργία στην πολιτική τους
5. Το MACF καλεί τις σχετικές πολιτικές
6. Οι πολιτικές υποδεικνύουν αν επιτρέπουν ή απορρίπτουν την ενέργεια

{% hint style="danger" %}
Η Apple είναι η μόνη που μπορεί να χρησιμοποιήσει το KPI του MAC Framework.
{% endhint %}

### Labels

Το MACF χρησιμοποιεί **ετικέτες** που στη συνέχεια οι πολιτικές ελέγχουν αν θα παραχωρήσουν κάποια πρόσβαση ή όχι. Ο κώδικας της δήλωσης δομής των ετικετών μπορεί να βρεθεί [εδώ](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h), ο οποίος χρησιμοποιείται στη συνέχεια μέσα στη **`struct ucred`** [**εδώ**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) στο μέρος **`cr_label`**. Η ετικέτα περιέχει σημαίες και έναν αριθμό **slots** που μπορούν να χρησιμοποιηθούν από **πολιτικές MACF για να εκχωρήσουν δείκτες**. Για παράδειγμα, το Sandbox θα δείχνει στο προφίλ του κοντέινερ.

## MACF Policies

Μια πολιτική MACF καθορίζει **κανόνες και συνθήκες που θα εφαρμοστούν σε ορισμένες λειτουργίες του πυρήνα**.&#x20;

Μια επέκταση πυρήνα θα μπορούσε να διαμορφώσει μια δομή `mac_policy_conf` και στη συνέχεια να την καταχωρίσει καλώντας `mac_policy_register`. Από [εδώ](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
Είναι εύκολο να εντοπιστούν οι επεκτάσεις πυρήνα που ρυθμίζουν αυτές τις πολιτικές ελέγχοντας τις κλήσεις προς το `mac_policy_register`. Επιπλέον, ελέγχοντας την αποσυναρμολόγηση της επέκτασης είναι επίσης δυνατό να βρεθεί η χρησιμοποιούμενη δομή `mac_policy_conf`.

Σημειώστε ότι οι πολιτικές MACF μπορούν να καταχωρηθούν και να αποσυρθούν επίσης **δυναμικά**.

Ένα από τα κύρια πεδία της `mac_policy_conf` είναι το **`mpc_ops`**. Αυτό το πεδίο καθορίζει ποιες λειτουργίες ενδιαφέρει η πολιτική. Σημειώστε ότι υπάρχουν εκατοντάδες από αυτές, οπότε είναι δυνατό να μηδενιστούν όλες και στη συνέχεια να επιλεγούν μόνο αυτές που ενδιαφέρουν την πολιτική. Από [εδώ](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
Σχεδόν όλοι οι hooks θα καλούνται από το MACF όταν μία από αυτές τις λειτουργίες παρεμποδίζεται. Ωστόσο, οι **`mpo_policy_*`** hooks είναι μια εξαίρεση επειδή το `mpo_hook_policy_init()` είναι μια callback που καλείται κατά την εγγραφή (οπότε μετά το `mac_policy_register()`) και το `mpo_hook_policy_initbsd()` καλείται κατά την καθυστερημένη εγγραφή μόλις το BSD subsystem έχει αρχικοποιηθεί σωστά.

Επιπλέον, ο **`mpo_policy_syscall`** hook μπορεί να εγγραφεί από οποιοδήποτε kext για να εκθέσει μια ιδιωτική **ioctl** στυλ κλήση **interface**. Στη συνέχεια, ένας πελάτης χρήστη θα μπορεί να καλέσει το `mac_syscall` (#381) καθορίζοντας ως παραμέτρους το **όνομα πολιτικής** με έναν ακέραιο **κωδικό** και προαιρετικά **ορίσματα**.\
Για παράδειγμα, το **`Sandbox.kext`** το χρησιμοποιεί πολύ.

Ελέγχοντας το **`__DATA.__const*`** του kext είναι δυνατό να προσδιοριστεί η δομή `mac_policy_ops` που χρησιμοποιείται κατά την εγγραφή της πολιτικής. Είναι δυνατό να την βρείτε επειδή ο δείκτης της είναι σε μια απόσταση μέσα στο `mpo_policy_conf` και επίσης λόγω του αριθμού των NULL δεικτών που θα είναι σε αυτήν την περιοχή.

Επιπλέον, είναι επίσης δυνατό να αποκτήσετε τη λίστα των kexts που έχουν ρυθμίσει μια πολιτική εκ dumping από τη μνήμη της δομής **`_mac_policy_list`** που ενημερώνεται με κάθε πολιτική που εγγράφεται.

## MACF Initialization

Το MACF αρχικοποιείται πολύ νωρίς. Ρυθμίζεται στο `bootstrap_thread` του XNU: μετά το `ipc_bootstrap` καλείται το `mac_policy_init()` που αρχικοποιεί τη λίστα `mac_policy_list` και λίγο αργότερα καλείται το `mac_policy_initmach()`. Μεταξύ άλλων, αυτή η λειτουργία θα αποκτήσει όλα τα Apple kexts με το κλειδί `AppleSecurityExtension` στο Info.plist τους όπως το `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` και `TMSafetyNet.kext` και τα φορτώνει.

## MACF Callouts

Είναι κοινό να βρείτε callouts στο MACF που ορίζονται σε κώδικα όπως: **`#if CONFIG_MAC`** μπλοκ συνθηκών. Επιπλέον, μέσα σε αυτά τα μπλοκ είναι δυνατό να βρείτε κλήσεις σε `mac_proc_check*` που καλεί το MACF για **έλεγχο δικαιωμάτων** για την εκτέλεση ορισμένων ενεργειών. Επιπλέον, η μορφή των callouts του MACF είναι: **`mac_<object>_<opType>_opName`**.

Το αντικείμενο είναι ένα από τα εξής: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
Ο `opType` είναι συνήθως check που θα χρησιμοποιηθεί για να επιτρέψει ή να αρνηθεί την ενέργεια. Ωστόσο, είναι επίσης δυνατό να βρείτε `notify`, που θα επιτρέψει στο kext να αντιδράσει στην δεδομένη ενέργεια.

Μπορείτε να βρείτε ένα παράδειγμα στο [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621):

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

Στη συνέχεια, είναι δυνατό να βρείτε τον κώδικα του `mac_file_check_mmap` στο [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174)
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
Ποιος καλεί το μακροεντολή `MAC_CHECK`, του οποίου ο κώδικας μπορεί να βρεθεί στο [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261)
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
Ποια θα περάσει από όλες τις καταχωρημένες πολιτικές mac καλώντας τις συναρτήσεις τους και αποθηκεύοντας την έξοδο μέσα στη μεταβλητή σφάλματος, η οποία θα μπορεί να παρακαμφθεί μόνο από το `mac_error_select` με κωδικούς επιτυχίας, έτσι ώστε αν οποιαδήποτε έλεγχος αποτύχει, ο συνολικός έλεγχος θα αποτύχει και η ενέργεια δεν θα επιτρέπεται.

{% hint style="success" %}
Ωστόσο, θυμηθείτε ότι δεν χρησιμοποιούνται όλες οι κλήσεις MACF μόνο για να αρνηθούν ενέργειες. Για παράδειγμα, το `mac_priv_grant` καλεί το μακροεντολή [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L274), η οποία θα παραχωρήσει το ζητούμενο προνόμιο αν οποιαδήποτε πολιτική απαντήσει με 0:
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

Αυτές οι κλήσεις προορίζονται να ελέγξουν και να παρέχουν (δεκάδες) **privileges** που ορίζονται στο [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h).\
Ορισμένος κώδικας πυρήνα θα καλεί το `priv_check_cred()` από [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) με τα KAuth credentials της διαδικασίας και έναν από τους κωδικούς privileges που θα καλέσει το `mac_priv_check` για να δει αν κάποια πολιτική **denies** την παροχή του privilege και στη συνέχεια καλεί το `mac_priv_grant` για να δει αν κάποια πολιτική παραχωρεί το `privilege`.

### proc\_check\_syscall\_unix

Αυτή η hook επιτρέπει την παρεμβολή σε όλες τις κλήσεις συστήματος. Στο `bsd/dev/[i386|arm]/systemcalls.c` είναι δυνατή η προβολή της δηλωμένης συνάρτησης [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25), η οποία περιέχει αυτόν τον κώδικα:
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
Ποιο θα ελέγξει στη διαδικασία κλήσης **bitmask** αν η τρέχουσα syscall θα πρέπει να καλέσει `mac_proc_check_syscall_unix`. Αυτό συμβαίνει επειδή οι syscalls καλούνται τόσο συχνά που είναι ενδιαφέρον να αποφευχθεί η κλήση του `mac_proc_check_syscall_unix` κάθε φορά.

Σημειώστε ότι η συνάρτηση `proc_set_syscall_filter_mask()`, η οποία ρυθμίζει το bitmask syscalls σε μια διαδικασία, καλείται από το Sandbox για να ρυθμίσει μάσκες σε διαδικασίες που είναι σε sandbox.

## Εκτεθειμένες syscalls MACF

Είναι δυνατόν να αλληλεπιδράσετε με το MACF μέσω ορισμένων syscalls που ορίζονται στο [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151):
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
## Αναφορές

* [**\*OS Internals Volume III**](https://newosxbook.com/home.html)

{% hint style="success" %}
Μάθετε & εξασκηθείτε στο AWS Hacking:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Μάθετε & εξασκηθείτε στο GCP Hacking: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Υποστήριξη HackTricks</summary>

* Ελέγξτε τα [**σχέδια συνδρομής**](https://github.com/sponsors/carlospolop)!
* **Εγγραφείτε στην** 💬 [**ομάδα Discord**](https://discord.gg/hRep4RUj7f) ή στην [**ομάδα telegram**](https://t.me/peass) ή **ακολουθήστε** μας στο **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Μοιραστείτε κόλπα hacking υποβάλλοντας PRs στα** [**HackTricks**](https://github.com/carlospolop/hacktricks) και [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repos.

</details>
{% endhint %}
