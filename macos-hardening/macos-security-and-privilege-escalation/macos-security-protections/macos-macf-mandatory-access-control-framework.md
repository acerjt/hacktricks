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

**MACF**는 **Mandatory Access Control Framework**의 약자로, 컴퓨터를 보호하기 위해 운영 체제에 내장된 보안 시스템입니다. 이는 **파일, 애플리케이션 및 시스템 리소스와 같은 시스템의 특정 부분에 접근할 수 있는 사람이나 사물에 대한 엄격한 규칙을 설정**하여 작동합니다. 이러한 규칙을 자동으로 시행함으로써, MACF는 권한이 있는 사용자와 프로세스만 특정 작업을 수행할 수 있도록 보장하여 무단 접근이나 악의적인 활동의 위험을 줄입니다.

MACF는 실제로 결정을 내리지 않고 **작업을 가로채기**만 하며, 결정을 내리는 것은 `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext` 및 `mcxalr.kext`와 같은 **정책 모듈**(커널 확장)에 맡깁니다.

### Flow

1. 프로세스가 syscall/mach trap을 수행합니다.
2. 관련 함수가 커널 내에서 호출됩니다.
3. 함수가 MACF를 호출합니다.
4. MACF는 해당 함수에 후킹을 요청한 정책 모듈을 확인합니다.
5. MACF는 관련 정책을 호출합니다.
6. 정책은 작업을 허용할지 거부할지를 나타냅니다.

{% hint style="danger" %}
Apple만이 MAC Framework KPI를 사용할 수 있습니다.
{% endhint %}

### Labels

MACF는 **접근을 허용할지 여부를 확인하는 정책에서 사용할 수 있는 레이블**을 사용합니다. 레이블 구조체 선언의 코드는 [여기](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/_label.h)에서 찾을 수 있으며, 이는 **`struct ucred`** 내의 [**여기**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/ucred.h#L86)에서 **`cr_label`** 부분에 사용됩니다. 레이블은 플래그와 **MACF 정책이 포인터를 할당하는 데 사용할 수 있는 슬롯의 수**를 포함합니다. 예를 들어, Sandbox는 컨테이너 프로필을 가리킵니다.

## MACF Policies

MACF 정책은 **특정 커널 작업에 적용될 규칙과 조건을 정의합니다**.&#x20;

커널 확장은 `mac_policy_conf` 구조체를 구성한 다음 `mac_policy_register`를 호출하여 등록할 수 있습니다. [여기](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html)에서:
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
커널 확장을 통해 이러한 정책을 구성하는 것을 식별하는 것은 `mac_policy_register` 호출을 확인하는 것으로 쉽게 할 수 있습니다. 또한, 확장의 디스어셈블을 확인하면 사용된 `mac_policy_conf` 구조체를 찾을 수 있습니다.

MACF 정책은 **동적으로** 등록 및 등록 해제할 수 있습니다.

`mac_policy_conf`의 주요 필드 중 하나는 **`mpc_ops`**입니다. 이 필드는 정책이 관심 있는 작업을 지정합니다. 수백 개가 있으므로 모든 작업을 0으로 설정한 다음 정책이 관심 있는 작업만 선택할 수 있습니다. [여기](https://opensource.apple.com/source/xnu/xnu-2050.18.24/security/mac_policy.h.auto.html)에서:
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
거의 모든 후크는 이러한 작업이 가로채질 때 MACF에 의해 호출됩니다. 그러나 **`mpo_policy_*`** 후크는 예외입니다. `mpo_hook_policy_init()`은 등록 시 호출되는 콜백이며(즉, `mac_policy_register()` 이후) `mpo_hook_policy_initbsd()`는 BSD 서브시스템이 제대로 초기화된 후 늦은 등록 중에 호출됩니다.

게다가, **`mpo_policy_syscall`** 후크는 모든 kext에 의해 등록될 수 있으며, 이를 통해 개인 **ioctl** 스타일 호출 **인터페이스**를 노출할 수 있습니다. 그러면 사용자 클라이언트는 **정책 이름**과 정수 **코드**, 선택적 **인수**를 매개변수로 지정하여 `mac_syscall` (#381)을 호출할 수 있습니다.\
예를 들어, **`Sandbox.kext`**는 이를 많이 사용합니다.

kext의 **`__DATA.__const*`**를 확인하면 정책 등록 시 사용되는 `mac_policy_ops` 구조체를 식별할 수 있습니다. 이는 `mpo_policy_conf` 내부의 오프셋에 포인터가 있기 때문에 찾을 수 있으며, 해당 영역에 있는 NULL 포인터의 수로도 찾을 수 있습니다.

또한, 메모리에서 구조체 **`_mac_policy_list`**를 덤프하여 정책을 구성한 kext의 목록을 얻는 것도 가능합니다. 이 구조체는 등록된 각 정책으로 업데이트됩니다.

## MACF 초기화

MACF는 매우 빨리 초기화됩니다. XNU의 `bootstrap_thread`에서 설정됩니다: `ipc_bootstrap` 이후 `mac_policy_init()` 호출이 이루어지며, 이는 `mac_policy_list`를 초기화하고 잠시 후 `mac_policy_initmach()`가 호출됩니다. 이 함수는 `ALF.kext`, `AppleMobileFileIntegrity.kext`, `Quarantine.kext`, `Sandbox.kext`, `TMSafetyNet.kext`와 같은 Info.plist에 `AppleSecurityExtension` 키가 있는 모든 Apple kext를 가져와서 로드합니다.

## MACF 호출

코드에서 **`#if CONFIG_MAC`** 조건부 블록과 같이 MACF에 대한 호출을 찾는 것은 일반적입니다. 또한, 이러한 블록 내에서 특정 작업을 수행하기 위한 권한을 **확인하기 위해** MACF를 호출하는 `mac_proc_check*` 호출을 찾을 수 있습니다. MACF 호출의 형식은 **`mac_<object>_<opType>_opName`**입니다.

객체는 다음 중 하나입니다: `bpfdesc`, `cred`, `file`, `proc`, `vnode`, `mount`, `devfs`, `ifnet`, `inpcb`, `mbuf`, `ipq`, `pipe`, `sysv[msg/msq/shm/sem]`, `posix[shm/sem]`, `socket`, `kext`.\
`opType`은 일반적으로 작업을 허용하거나 거부하는 데 사용되는 check입니다. 그러나 kext가 주어진 작업에 반응할 수 있도록 하는 notify를 찾는 것도 가능합니다.

[https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/kern/kern\_mman.c#L621)에서 예제를 찾을 수 있습니다:

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

그런 다음 [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_file.c#L174)에서 `mac_file_check_mmap`의 코드를 찾을 수 있습니다.
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
`MAC_CHECK` 매크로를 호출하고 있으며, 해당 코드는 [https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L261)에서 찾을 수 있습니다.
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
어떤 등록된 mac 정책을 호출하고 그 함수의 출력을 error 변수에 저장하는데, 이 변수는 성공 코드에 의해 `mac_error_select`로만 재정의될 수 있습니다. 따라서 어떤 체크가 실패하면 전체 체크가 실패하고 액션이 허용되지 않습니다.

{% hint style="success" %}
그러나 모든 MACF 호출이 액션을 거부하는 데만 사용되는 것은 아니라는 점을 기억하세요. 예를 들어, `mac_priv_grant`는 매크로 [**MAC\_GRANT**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac\_internal.h#L274)를 호출하며, 이는 어떤 정책이 0으로 응답하면 요청된 권한을 부여합니다:
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

이 호출은 [**bsd/sys/priv.h**](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/sys/priv.h)에서 정의된 (수십 개의) **권한**을 확인하고 제공하기 위한 것입니다.\
일부 커널 코드는 프로세스의 KAuth 자격 증명과 함께 `priv_check_cred()`를 호출하여 권한을 부여하는 정책이 **거부**되는지 확인하기 위해 `mac_priv_check`를 호출하고, 그런 다음 `privilege`를 부여하는 정책이 있는지 확인하기 위해 `mac_priv_grant`를 호출합니다.

### proc\_check\_syscall\_unix

이 후크는 모든 시스템 호출을 가로챌 수 있게 해줍니다. `bsd/dev/[i386|arm]/systemcalls.c`에서 선언된 함수 [`unix_syscall`](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/dev/arm/systemcalls.c#L160C1-L167C25)를 확인할 수 있으며, 이 코드가 포함되어 있습니다:
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
어떤 것이 호출 프로세스의 **비트마스크**에서 현재 시스템 호출이 `mac_proc_check_syscall_unix`를 호출해야 하는지를 확인합니다. 이는 시스템 호출이 매우 자주 호출되기 때문에 매번 `mac_proc_check_syscall_unix`를 호출하는 것을 피하는 것이 흥미롭기 때문입니다.

`proc_set_syscall_filter_mask()` 함수는 프로세스의 비트마스크 시스템 호출을 설정하며, Sandbox에 의해 샌드박스화된 프로세스에 마스크를 설정하기 위해 호출됩니다.

## 노출된 MACF 시스템 호출

[security/mac.h](https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/security/mac.h#L151)에서 정의된 일부 시스템 호출을 통해 MACF와 상호작용하는 것이 가능합니다:
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
Learn & practice AWS Hacking:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Learn & practice GCP Hacking: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Support HackTricks</summary>

* Check the [**subscription plans**](https://github.com/sponsors/carlospolop)!
* **Join the** 💬 [**Discord group**](https://discord.gg/hRep4RUj7f) or the [**telegram group**](https://t.me/peass) or **follow** us on **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Share hacking tricks by submitting PRs to the** [**HackTricks**](https://github.com/carlospolop/hacktricks) and [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repos.

</details>
{% endhint %}
