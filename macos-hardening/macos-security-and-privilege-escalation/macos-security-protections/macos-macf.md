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

**MACF** significa **Mandatory Access Control Framework**, que é um sistema de segurança embutido no sistema operacional para ajudar a proteger seu computador. Ele funciona estabelecendo **regras rigorosas sobre quem ou o que pode acessar certas partes do sistema**, como arquivos, aplicativos e recursos do sistema. Ao impor essas regras automaticamente, o MACF garante que apenas usuários e processos autorizados possam realizar ações específicas, reduzindo o risco de acesso não autorizado ou atividades maliciosas.

Observe que o MACF não toma realmente decisões, pois apenas **intercepta** ações, deixando as decisões para os **módulos de política** (extensões do kernel) que chama, como `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` e `mcxalr.kext`.

### Flow

1. O processo realiza uma syscall/mach trap
2. A função relevante é chamada dentro do kernel
3. A função chama o MACF
4. O MACF verifica os módulos de política que solicitaram para interceptar essa função em sua política
5. O MACF chama as políticas relevantes
6. As políticas indicam se permitem ou negam a ação

{% hint style="danger" %}
A Apple é a única que pode usar o KPI do MAC Framework.
{% endhint %}

### Labels

O MACF usa **labels** que, em seguida, as políticas verificam se devem conceder algum acesso ou não. O código da declaração da estrutura de labels pode ser [encontrado aqui](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h), que é então usado dentro da **`struct ucred`** em [**aqui**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) na parte **`cr_label`**. A label contém flags e um número de **slots** que podem ser usados pelas **políticas MACF para alocar ponteiros**. Por exemplo, o Sandbox apontará para o perfil do contêiner.

## MACF Policies

Uma Política MACF define **regras e condições a serem aplicadas em certas operações do kernel**.&#x20;

Uma extensão do kernel poderia configurar uma estrutura `mac_policy_conf` e, em seguida, registrá-la chamando `mac_policy_register`. A partir [daqui](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
É fácil identificar as extensões do kernel que configuram essas políticas verificando chamadas para `mac_policy_register`. Além disso, verificando a desassemblagem da extensão, também é possível encontrar a struct `mac_policy_conf` utilizada.

Observe que as políticas MACF podem ser registradas e desregistradas também **dinamicamente**.

Um dos principais campos da `mac_policy_conf` é o **`mpc_ops`**. Este campo especifica quais operações a política está interessada. Note que existem centenas delas, então é possível zerar todas e, em seguida, selecionar apenas aquelas que a política está interessada. De [aqui](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac\_policy.h.auto.html):
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
Quase todos os hooks serão chamados de volta pelo MACF quando uma dessas operações for interceptada. No entanto, os hooks **`mpo_policy_*`** são uma exceção porque `mpo_hook_policy_init()` é um callback chamado durante o registro (após `mac_policy_register()`) e `mpo_hook_policy_initbsd()` é chamado durante o registro tardio, uma vez que o subsistema BSD foi inicializado corretamente.

Além disso, o hook **`mpo_policy_syscall`** pode ser registrado por qualquer kext para expor uma chamada de estilo **ioctl** **interface** privada. Então, um cliente de usuário poderá chamar `mac_syscall` (#381) especificando como parâmetros o **nome da política** com um **código** inteiro e **argumentos** opcionais.\
Por exemplo, o **`Sandbox.kext`** usa isso com frequência.

Verificando o **`__DATA.__const*`** do kext, é possível identificar a estrutura `mac_policy_ops` usada ao registrar a política. É possível encontrá-la porque seu ponteiro está em um deslocamento dentro de `mpo_policy_conf` e também devido à quantidade de ponteiros NULL que estarão naquela área.

Além disso, também é possível obter a lista de kexts que configuraram uma política despejando da memória a estrutura **`_mac_policy_list`** que é atualizada com cada política que é registrada.

## Inicialização do MACF

O MACF é inicializado muito cedo. Ele é configurado na `bootstrap_thread` do XNU: após `ipc_bootstrap`, uma chamada para `mac_policy_init()` que inicializa a `mac_policy_list` e momentos depois `mac_policy_initmach()` é chamado. Entre outras coisas, essa função obterá todos os kexts da Apple com a chave `AppleSecurityExtension` em seu Info.plist, como `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` e `TMSafetyNet.kext` e os carrega.

## Chamadas do MACF

É comum encontrar chamadas para o MACF definidas em código como: **`#if CONFIG_MAC`** blocos condicionais. Além disso, dentro desses blocos, é possível encontrar chamadas para `mac_proc_check*` que chamam o MACF para **verificar permissões** para realizar certas ações. Além disso, o formato das chamadas do MACF é: **`mac_<object>_<opType>_opName`**.

O objeto é um dos seguintes: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
O `opType` geralmente é check, que será usado para permitir ou negar a ação. No entanto, também é possível encontrar notify, que permitirá que o kext reaja à ação dada.

Você pode encontrar um exemplo em [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621):

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

Então, é possível encontrar o código de `mac_file_check_mmap` em [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174)
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
Qual está chamando o macro `MAC_CHECK`, cujo código pode ser encontrado em [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261)
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
Qualquer um que irá percorrer todas as políticas mac registradas chamando suas funções e armazenando a saída dentro da variável de erro, que só poderá ser substituída por `mac_error_select` por códigos de sucesso, então se qualquer verificação falhar, a verificação completa falhará e a ação não será permitida.

{% hint style="success" %}
No entanto, lembre-se de que nem todos os callouts MACF são usados apenas para negar ações. Por exemplo, `mac_priv_grant` chama o macro [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac_internal.h#L274), que concederá o privilégio solicitado se qualquer política responder com um 0:
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

Essas chamadas são destinadas a verificar e fornecer (dezenas de) **privilegios** definidos em [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h).\
Algum código do kernel chamaria `priv_check_cred()` de [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) com as credenciais KAuth do processo e um dos códigos de privilégio que chamará `mac_priv_check` para ver se alguma política **nega** a concessão do privilégio e, em seguida, chama `mac_priv_grant` para ver se alguma política concede o `privilegio`.

### proc\_check\_syscall\_unix

Esse hook permite interceptar todas as chamadas de sistema. Em `bsd/dev/[i386|arm]/systemcalls.c` é possível ver a função declarada [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25), que contém este código:
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
Qual verificará no **bitmask** do processo chamador se a syscall atual deve chamar `mac_proc_check_syscall_unix`. Isso ocorre porque as syscalls são chamadas com tanta frequência que é interessante evitar chamar `mac_proc_check_syscall_unix` toda vez.

Observe que a função `proc_set_syscall_filter_mask()`, que define o bitmask das syscalls em um processo, é chamada pelo Sandbox para definir máscaras em processos isolados.

## Syscalls MACF expostas

É possível interagir com o MACF através de algumas syscalls definidas em [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151):
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
## Referências

* [**\*OS Internals Volume III**](https://newosxbook.com/home.html)

{% hint style="success" %}
Aprenda e pratique Hacking AWS:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Aprenda e pratique Hacking GCP: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Support HackTricks</summary>

* Confira os [**planos de assinatura**](https://github.com/sponsors/carlospolop)!
* **Junte-se ao** 💬 [**grupo do Discord**](https://discord.gg/hRep4RUj7f) ou ao [**grupo do telegram**](https://t.me/peass) ou **siga**-nos no **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Compartilhe truques de hacking enviando PRs para os repositórios do** [**HackTricks**](https://github.com/carlospolop/hacktricks) e [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud).

</details>
{% endhint %}
