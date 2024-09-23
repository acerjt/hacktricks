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



## AppleMobileFileIntegrity.kext y amfid

Se centra en hacer cumplir la integridad del código que se ejecuta en el sistema, proporcionando la lógica detrás de la verificación de la firma de código de XNU. También es capaz de verificar derechos y manejar otras tareas sensibles, como permitir la depuración u obtener puertos de tarea.

Además, para algunas operaciones, el kext prefiere contactar al espacio de usuario ejecutando el daemon `/usr/libexec/amfid`. Esta relación de confianza ha sido abusada en varios jailbreaks.

AMFI utiliza políticas **MACF** y registra sus hooks en el momento en que se inicia. Además, prevenir su carga o descarga podría desencadenar un pánico del kernel. Sin embargo, hay algunos argumentos de arranque que permiten debilitar AMFI:

* `amfi_unrestricted_task_for_pid`: Permitir que task\_for\_pid sea permitido sin derechos requeridos
* `amfi_allow_any_signature`: Permitir cualquier firma de código
* `cs_enforcement_disable`: Argumento a nivel de sistema utilizado para deshabilitar la aplicación de la firma de código
* `amfi_prevent_old_entitled_platform_binaries`: Anular binarios de plataforma con derechos
* `amfi_get_out_of_my_way`: Deshabilita amfi completamente

Estas son algunas de las políticas MACF que registra:

* **`cred_check_label_update_execve:`** Se realizará una actualización de etiqueta y devolverá 1
* **`cred_label_associate`**: Actualiza el slot de etiqueta mac de AMFI con la etiqueta
* **`cred_label_destroy`**: Elimina el slot de etiqueta mac de AMFI
* **`cred_label_init`**: Mueve 0 en el slot de etiqueta mac de AMFI
* **`cred_label_update_execve`:** Verifica los derechos del proceso para ver si se le debe permitir modificar las etiquetas.
* **`file_check_mmap`:** Verifica si mmap está adquiriendo memoria y configurándola como ejecutable. En ese caso, verifica si se necesita validación de biblioteca y, si es así, llama a la función de validación de biblioteca.
* **`file_check_library_validation`**: Llama a la función de validación de biblioteca que verifica, entre otras cosas, si un binario de plataforma está cargando otro binario de plataforma o si el proceso y el nuevo archivo cargado tienen el mismo TeamID. Ciertos derechos también permitirán cargar cualquier biblioteca.
* **`policy_initbsd`**: Configura claves NVRAM de confianza
* **`policy_syscall`**: Verifica políticas DYLD como si el binario tiene segmentos sin restricciones, si debe permitir variables de entorno... esto también se llama cuando un proceso se inicia a través de `amfi_check_dyld_policy_self()`.
* **`proc_check_inherit_ipc_ports`**: Verifica si cuando un proceso ejecuta un nuevo binario, otros procesos con derechos de ENVÍO sobre el puerto de tarea del proceso deben mantenerlos o no. Se permiten binarios de plataforma, el derecho `get-task-allow` lo permite, los derechos `task_for_pid-allow` son permitidos y los binarios con el mismo TeamID.
* **`proc_check_expose_task`**: hace cumplir los derechos
* **`amfi_exc_action_check_exception_send`**: Se envía un mensaje de excepción al depurador
* **`amfi_exc_action_label_associate & amfi_exc_action_label_copy/populate & amfi_exc_action_label_destroy & amfi_exc_action_label_init & amfi_exc_action_label_update`**: Ciclo de vida de la etiqueta durante el manejo de excepciones (depuración)
* **`proc_check_get_task`**: Verifica derechos como `get-task-allow` que permite a otros procesos obtener el puerto de tareas y `task_for_pid-allow`, que permite al proceso obtener los puertos de tareas de otros procesos. Si ninguno de esos, llama a `amfid permitunrestricteddebugging` para verificar si está permitido.
* **`proc_check_mprotect`**: Niega si `mprotect` se llama con la bandera `VM_PROT_TRUSTED`, que indica que la región debe ser tratada como si tuviera una firma de código válida.
* **`vnode_check_exec`**: Se llama cuando se cargan archivos ejecutables en memoria y establece `cs_hard | cs_kill`, que matará el proceso si alguna de las páginas se vuelve inválida
* **`vnode_check_getextattr`**: MacOS: Verifica `com.apple.root.installed` y `isVnodeQuarantined()`
* **`vnode_check_setextattr`**: Como obtener + com.apple.private.allow-bless y derecho equivalente de instalador interno
* &#x20;**`vnode_check_signature`**: Código que llama a XNU para verificar la firma de código utilizando derechos, caché de confianza y `amfid`
* &#x20;**`proc_check_run_cs_invalid`**: Intercepta llamadas a `ptrace()` (`PT_ATTACH` y `PT_TRACE_ME`). Verifica si alguno de los derechos `get-task-allow`, `run-invalid-allow` y `run-unsigned-code` y si ninguno, verifica si se permite la depuración.
* **`proc_check_map_anon`**: Si mmap se llama con la bandera **`MAP_JIT`**, AMFI verificará el derecho `dynamic-codesigning`.

`AMFI.kext` también expone una API para otras extensiones del kernel, y es posible encontrar sus dependencias con:
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

Este es el demonio que se ejecuta en modo usuario que `AMFI.kext` utilizará para verificar las firmas de código en modo usuario.\
Para que `AMFI.kext` se comunique con el demonio, utiliza mensajes mach a través del puerto `HOST_AMFID_PORT`, que es el puerto especial `18`.

Ten en cuenta que en macOS ya no es posible que los procesos root secuestren puertos especiales, ya que están protegidos por `SIP` y solo launchd puede acceder a ellos. En iOS se verifica que el proceso que envía la respuesta tenga el CDHash codificado de `amfid`.

Es posible ver cuándo se solicita a `amfid` que verifique un binario y la respuesta de este depurándolo y estableciendo un punto de interrupción en `mach_msg`.

Una vez que se recibe un mensaje a través del puerto especial, **MIG** se utiliza para enviar cada función a la función que está llamando. Las funciones principales fueron revertidas y explicadas dentro del libro.

## Provisioning Profiles

Un perfil de aprovisionamiento se puede utilizar para firmar código. Hay perfiles de **Desarrollador** que se pueden utilizar para firmar código y probarlo, y perfiles **Empresariales** que se pueden utilizar en todos los dispositivos.

Después de que una aplicación se envía a la Apple Store, si es aprobada, es firmada por Apple y el perfil de aprovisionamiento ya no es necesario.

Un perfil generalmente utiliza la extensión `.mobileprovision` o `.provisionprofile` y se puede volcar con:
```bash
openssl asn1parse -inform der -in /path/to/profile

# Or

security cms -D -i /path/to/profile
```
Aunque a veces se les llama certificados, estos perfiles de aprovisionamiento tienen más que un certificado:

* **AppIDName:** El Identificador de Aplicación
* **AppleInternalProfile**: Designa esto como un perfil interno de Apple
* **ApplicationIdentifierPrefix**: Precedido al AppIDName (igual que TeamIdentifier)
* **CreationDate**: Fecha en formato `YYYY-MM-DDTHH:mm:ssZ`
* **DeveloperCertificates**: Un array de (usualmente uno) certificado(s), codificado como datos Base64
* **Entitlements**: Los derechos permitidos con derechos para este perfil
* **ExpirationDate**: Fecha de expiración en formato `YYYY-MM-DDTHH:mm:ssZ`
* **Name**: El Nombre de la Aplicación, el mismo que AppIDName
* **ProvisionedDevices**: Un array (para certificados de desarrollador) de UDIDs para los que este perfil es válido
* **ProvisionsAllDevices**: Un booleano (verdadero para certificados empresariales)
* **TeamIdentifier**: Un array de (usualmente uno) cadena(s) alfanumérica(s) utilizadas para identificar al desarrollador para propósitos de interacción entre aplicaciones
* **TeamName**: Un nombre legible por humanos utilizado para identificar al desarrollador
* **TimeToLive**: Validez (en días) del certificado
* **UUID**: Un Identificador Único Universal para este perfil
* **Version**: Actualmente establecido en 1

Tenga en cuenta que la entrada de derechos contendrá un conjunto restringido de derechos y el perfil de aprovisionamiento solo podrá otorgar esos derechos específicos para evitar otorgar derechos privados de Apple.

Tenga en cuenta que los perfiles generalmente se encuentran en `/var/MobileDeviceProvisioningProfiles` y es posible verificarlos con **`security cms -D -i /path/to/profile`**

## **libmis.dyld**

Esta es la biblioteca externa que `amfid` llama para preguntar si debe permitir algo o no. Esto ha sido abusado históricamente en el jailbreak ejecutando una versión con puerta trasera que permitiría todo.

En macOS esto está dentro de `MobileDevice.framework`.

## AMFI Trust Caches

iOS AMFI mantiene una lista de hashes conocidos que están firmados ad-hoc, llamada **Trust Cache** y se encuentra en la sección `__TEXT.__const` del kext. Tenga en cuenta que en operaciones muy específicas y sensibles es posible extender esta Trust Cache con un archivo externo.

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
