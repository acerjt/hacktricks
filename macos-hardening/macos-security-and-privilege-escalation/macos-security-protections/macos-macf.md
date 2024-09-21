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

## Información Básica

**MACF** significa **Marco de Control de Acceso Obligatorio**, que es un sistema de seguridad integrado en el sistema operativo para ayudar a proteger tu computadora. Funciona estableciendo **reglas estrictas sobre quién o qué puede acceder a ciertas partes del sistema**, como archivos, aplicaciones y recursos del sistema. Al hacer cumplir estas reglas automáticamente, MACF asegura que solo los usuarios y procesos autorizados puedan realizar acciones específicas, reduciendo el riesgo de acceso no autorizado o actividades maliciosas.

Ten en cuenta que MACF realmente no toma decisiones, ya que solo **intercepta** acciones, deja las decisiones a los **módulos de política** (extensiones del kernel) que llama como `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` y `mcxalr.kext`.

### Flujo

1. El proceso realiza una llamada al syscall/trampa mach
2. Se llama a la función relevante dentro del kernel
3. La función llama a MACF
4. MACF verifica los módulos de política que solicitaron enganchar esa función en su política
5. MACF llama a las políticas relevantes
6. Las políticas indican si permiten o deniegan la acción

{% hint style="danger" %}
Apple es el único que puede usar el KPI del Marco MAC.
{% endhint %}

### Etiquetas

MACF utiliza **etiquetas** que luego las políticas comprobarán si deben otorgar algún acceso o no. El código de la declaración de la estructura de etiquetas se puede [encontrar aquí](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h), que se utiliza dentro de la **`struct ucred`** en [**aquí**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86) en la parte de **`cr_label`**. La etiqueta contiene banderas y un número de **slots** que pueden ser utilizados por **las políticas de MACF para asignar punteros**. Por ejemplo, Sanbox apuntará al perfil del contenedor.

## Políticas de MACF

Una Política de MACF define **reglas y condiciones que se aplicarán en ciertas operaciones del kernel**.&#x20;

Una extensión del kernel podría configurar una estructura `mac_policy_conf` y luego registrarla llamando a `mac_policy_register`. Desde [aquí](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html):
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
Es fácil identificar las extensiones del kernel que configuran estas políticas al verificar las llamadas a `mac_policy_register`. Además, al revisar el desensamblado de la extensión, también es posible encontrar la estructura `mac_policy_conf` utilizada.

Tenga en cuenta que las políticas MACF también se pueden registrar y anular **dinámicamente**.

Uno de los principales campos de `mac_policy_conf` es **`mpc_ops`**. Este campo especifica en qué operaciones está interesada la política. Tenga en cuenta que hay cientos de ellas, por lo que es posible establecer todas en cero y luego seleccionar solo las que le interesan a la política. Desde [aquí](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac\_policy.h.auto.html):
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
Casi todos los hooks serán llamados por MACF cuando una de esas operaciones sea interceptada. Sin embargo, los hooks **`mpo_policy_*`** son una excepción porque `mpo_hook_policy_init()` es un callback llamado al registrarse (después de `mac_policy_register()`) y `mpo_hook_policy_initbsd()` se llama durante el registro tardío una vez que el subsistema BSD se ha inicializado correctamente.

Además, el hook **`mpo_policy_syscall`** puede ser registrado por cualquier kext para exponer una llamada de estilo **ioctl** **interface** privada. Luego, un cliente de usuario podrá llamar a `mac_syscall` (#381) especificando como parámetros el **nombre de la política** con un **código** entero y **argumentos** opcionales.\
Por ejemplo, **`Sandbox.kext`** utiliza esto mucho.

Revisando el **`__DATA.__const*`** del kext es posible identificar la estructura `mac_policy_ops` utilizada al registrar la política. Es posible encontrarla porque su puntero está en un desplazamiento dentro de `mpo_policy_conf` y también por la cantidad de punteros NULL que habrá en esa área.

Además, también es posible obtener la lista de kexts que han configurado una política volcando de la memoria la estructura **`_mac_policy_list`** que se actualiza con cada política que se registra.

## Inicialización de MACF

MACF se inicializa muy pronto. Se configura en el `bootstrap_thread` de XNU: después de `ipc_bootstrap` se llama a `mac_policy_init()` que inicializa la `mac_policy_list` y momentos después se llama a `mac_policy_initmach()`. Entre otras cosas, esta función obtendrá todos los kexts de Apple con la clave `AppleSecurityExtension` en su Info.plist como `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext` y `TMSafetyNet.kext` y los carga.

## Llamadas de MACF

Es común encontrar llamadas a MACF definidas en el código como: bloques condicionales **`#if CONFIG_MAC`**. Además, dentro de estos bloques es posible encontrar llamadas a `mac_proc_check*` que llama a MACF para **verificar permisos** para realizar ciertas acciones. Además, el formato de las llamadas de MACF es: **`mac_<object>_<opType>_opName`**.

El objeto es uno de los siguientes: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
El `opType` suele ser check que se utilizará para permitir o denegar la acción. Sin embargo, también es posible encontrar `notify`, que permitirá al kext reaccionar a la acción dada.

Puedes encontrar un ejemplo en [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621):

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

Luego, es posible encontrar el código de `mac_file_check_mmap` en [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174)
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
El cual está llamando al macro `MAC_CHECK`, cuyo código se puede encontrar en [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261)
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
Lo que revisará todas las políticas mac registradas llamando a sus funciones y almacenando la salida dentro de la variable de error, que solo será sobreescribible por `mac_error_select` mediante códigos de éxito, por lo que si alguna verificación falla, la verificación completa fallará y la acción no será permitida.

{% hint style="success" %}
Sin embargo, recuerda que no todos los llamados de MACF se utilizan solo para denegar acciones. Por ejemplo, `mac_priv_grant` llama al macro [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L274), que otorgará el privilegio solicitado si alguna política responde con un 0:
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

Estas llamadas están destinadas a verificar y proporcionar (decenas de) **privilegios** definidos en [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h).\
Algunos códigos del kernel llamarían a `priv_check_cred()` desde [**bsd/kern/kern\_priv.c**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_priv.c) con las credenciales KAuth del proceso y uno de los códigos de privilegios que llamará a `mac_priv_check` para ver si alguna política **niega** otorgar el privilegio y luego llama a `mac_priv_grant` para ver si alguna política otorga el `privilegio`.

### proc\_check\_syscall\_unix

Este gancho permite interceptar todas las llamadas al sistema. En `bsd/dev/[i386|arm]/systemcalls.c` es posible ver la función declarada [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25), que contiene este código:
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
Que verificará en el proceso que llama **bitmask** si la syscall actual debería llamar a `mac_proc_check_syscall_unix`. Esto se debe a que las syscalls se llaman con tanta frecuencia que es interesante evitar llamar a `mac_proc_check_syscall_unix` cada vez.

Tenga en cuenta que la función `proc_set_syscall_filter_mask()`, que establece la máscara de syscalls en un proceso, es llamada por Sandbox para establecer máscaras en procesos en sandbox.

## Syscalls MACF expuestas

Es posible interactuar con MACF a través de algunas syscalls definidas en [security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151):
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
## Referencias

* [**\*OS Internals Volume III**](https://newosxbook.com/home.html)

{% hint style="success" %}
Aprende y practica Hacking en AWS:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Aprende y practica Hacking en GCP: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Apoya a HackTricks</summary>

* Revisa los [**planes de suscripción**](https://github.com/sponsors/carlospolop)!
* **Únete al** 💬 [**grupo de Discord**](https://discord.gg/hRep4RUj7f) o al [**grupo de telegram**](https://t.me/peass) o **síguenos** en **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Comparte trucos de hacking enviando PRs a los** [**HackTricks**](https://github.com/carlospolop/hacktricks) y [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) repositorios de github.

</details>
{% endhint %}
