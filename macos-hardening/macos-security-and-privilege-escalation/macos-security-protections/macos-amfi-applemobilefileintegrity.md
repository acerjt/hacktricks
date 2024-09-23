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



## AppleMobileFileIntegrity.kext e amfid

Si concentra sull'applicazione dell'integrità del codice in esecuzione sul sistema fornendo la logica dietro la verifica della firma del codice di XNU. È anche in grado di controllare i diritti e gestire altre operazioni sensibili come consentire il debug o ottenere porte di task.

Inoltre, per alcune operazioni, il kext preferisce contattare il demone in user space `/usr/libexec/amfid`. Questa relazione di fiducia è stata abusata in diversi jailbreak.

AMFI utilizza politiche **MACF** e registra i suoi hook nel momento in cui viene avviato. Inoltre, impedire il suo caricamento o scaricamento potrebbe innescare un kernel panic. Tuttavia, ci sono alcuni argomenti di avvio che consentono di indebolire AMFI:

* `amfi_unrestricted_task_for_pid`: Consente a task\_for\_pid di essere consentito senza diritti richiesti
* `amfi_allow_any_signature`: Consente qualsiasi firma del codice
* `cs_enforcement_disable`: Argomento a livello di sistema utilizzato per disabilitare l'applicazione della firma del codice
* `amfi_prevent_old_entitled_platform_binaries`: Annulla i binari di piattaforma con diritti
* `amfi_get_out_of_my_way`: Disabilita completamente amfi

Queste sono alcune delle politiche MACF che registra:

* **`cred_check_label_update_execve:`** L'aggiornamento dell'etichetta verrà eseguito e restituirà 1
* **`cred_label_associate`**: Aggiorna lo slot dell'etichetta mac di AMFI con l'etichetta
* **`cred_label_destroy`**: Rimuove lo slot dell'etichetta mac di AMFI
* **`cred_label_init`**: Sposta 0 nello slot dell'etichetta mac di AMFI
* **`cred_label_update_execve`:** Controlla i diritti del processo per vedere se dovrebbe essere consentito modificare le etichette.
* **`file_check_mmap`:** Controlla se mmap sta acquisendo memoria e impostandola come eseguibile. In tal caso, verifica se è necessaria la convalida della libreria e, in tal caso, chiama la funzione di convalida della libreria.
* **`file_check_library_validation`**: Chiama la funzione di convalida della libreria che controlla, tra le altre cose, se un binario di piattaforma sta caricando un altro binario di piattaforma o se il processo e il nuovo file caricato hanno lo stesso TeamID. Alcuni diritti consentiranno anche di caricare qualsiasi libreria.
* **`policy_initbsd`**: Configura le chiavi NVRAM fidate
* **`policy_syscall`**: Controlla le politiche DYLD come se il binario ha segmenti non limitati, se dovrebbe consentire le variabili d'ambiente... questo viene chiamato anche quando un processo viene avviato tramite `amfi_check_dyld_policy_self()`.
* **`proc_check_inherit_ipc_ports`**: Controlla se quando un processo esegue un nuovo binario altri processi con diritti SEND sulla porta del task del processo dovrebbero mantenerli o meno. I binari di piattaforma sono consentiti, il diritto `get-task-allow` lo consente, i diritti `task_for_pid-allow` sono consentiti e i binari con lo stesso TeamID.
* **`proc_check_expose_task`**: applica i diritti
* **`amfi_exc_action_check_exception_send`**: Un messaggio di eccezione viene inviato al debugger
* **`amfi_exc_action_label_associate & amfi_exc_action_label_copy/populate & amfi_exc_action_label_destroy & amfi_exc_action_label_init & amfi_exc_action_label_update`**: Ciclo di vita dell'etichetta durante la gestione delle eccezioni (debugging)
* **`proc_check_get_task`**: Controlla i diritti come `get-task-allow` che consente ad altri processi di ottenere la porta dei task e `task_for_pid-allow`, che consente al processo di ottenere le porte dei task di altri processi. Se nessuno di questi, chiama `amfid permitunrestricteddebugging` per controllare se è consentito.
* **`proc_check_mprotect`**: Negato se `mprotect` viene chiamato con il flag `VM_PROT_TRUSTED` che indica che la regione deve essere trattata come se avesse una firma del codice valida.
* **`vnode_check_exec`**: Viene chiamato quando i file eseguibili vengono caricati in memoria e imposta `cs_hard | cs_kill` che ucciderà il processo se una delle pagine diventa non valida
* **`vnode_check_getextattr`**: MacOS: Controlla `com.apple.root.installed` e `isVnodeQuarantined()`
* **`vnode_check_setextattr`**: Come get + com.apple.private.allow-bless e diritto equivalente di internal-installer
* &#x20;**`vnode_check_signature`**: Codice che chiama XNU per controllare la firma del codice utilizzando diritti, cache di fiducia e `amfid`
* &#x20;**`proc_check_run_cs_invalid`**: Intercetta le chiamate `ptrace()` (`PT_ATTACH` e `PT_TRACE_ME`). Controlla per eventuali diritti `get-task-allow`, `run-invalid-allow` e `run-unsigned-code` e se nessuno, verifica se il debug è consentito.
* **`proc_check_map_anon`**: Se mmap viene chiamato con il flag **`MAP_JIT`**, AMFI controllerà il diritto `dynamic-codesigning`.

`AMFI.kext` espone anche un'API per altre estensioni del kernel, ed è possibile trovare le sue dipendenze con:
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

Questo è il demone in modalità utente che `AMFI.kext` utilizzerà per controllare le firme del codice in modalità utente.\
Per consentire a `AMFI.kext` di comunicare con il demone, utilizza messaggi mach attraverso la porta `HOST_AMFID_PORT`, che è la porta speciale `18`.

Si noti che in macOS non è più possibile per i processi root di dirottare porte speciali poiché sono protette da `SIP` e solo launchd può accedervi. In iOS viene verificato che il processo che invia la risposta abbia il CDHash hardcoded di `amfid`.

È possibile vedere quando `amfid` viene richiesto di controllare un binario e la sua risposta eseguendo il debug e impostando un breakpoint in `mach_msg`.

Una volta ricevuto un messaggio tramite la porta speciale, **MIG** viene utilizzato per inviare ogni funzione alla funzione che sta chiamando. Le funzioni principali sono state analizzate e spiegate all'interno del libro.

## Provisioning Profiles

Un provisioning profile può essere utilizzato per firmare il codice. Ci sono profili **Developer** che possono essere utilizzati per firmare il codice e testarlo, e profili **Enterprise** che possono essere utilizzati su tutti i dispositivi.

Dopo che un'app è stata inviata all'Apple Store, se approvata, viene firmata da Apple e il provisioning profile non è più necessario.

Un profilo di solito utilizza l'estensione `.mobileprovision` o `.provisionprofile` e può essere estratto con:
```bash
openssl asn1parse -inform der -in /path/to/profile

# Or

security cms -D -i /path/to/profile
```
Sebbene a volte siano chiamati certificati, questi profili di provisioning hanno più di un certificato:

* **AppIDName:** L'Identificatore dell'Applicazione
* **AppleInternalProfile**: Designa questo come un profilo Interno Apple
* **ApplicationIdentifierPrefix**: Preceduto da AppIDName (stesso di TeamIdentifier)
* **CreationDate**: Data nel formato `YYYY-MM-DDTHH:mm:ssZ`
* **DeveloperCertificates**: Un array di (di solito uno) certificato(i), codificato come dati Base64
* **Entitlements**: I diritti consentiti con i diritti per questo profilo
* **ExpirationDate**: Data di scadenza nel formato `YYYY-MM-DDTHH:mm:ssZ`
* **Name**: Il Nome dell'Applicazione, lo stesso di AppIDName
* **ProvisionedDevices**: Un array (per certificati di sviluppatore) di UDID per cui questo profilo è valido
* **ProvisionsAllDevices**: Un booleano (true per certificati aziendali)
* **TeamIdentifier**: Un array di (di solito uno) stringhe alfanumeriche utilizzate per identificare lo sviluppatore per scopi di interazione tra app
* **TeamName**: Un nome leggibile dall'uomo utilizzato per identificare lo sviluppatore
* **TimeToLive**: Validità (in giorni) del certificato
* **UUID**: Un Identificatore Universale Unico per questo profilo
* **Version**: Attualmente impostato su 1

Nota che l'entry degli entitlements conterrà un insieme ristretto di diritti e il profilo di provisioning sarà in grado di fornire solo quegli specifici diritti per prevenire la concessione di diritti privati ad Apple.

Nota che i profili si trovano solitamente in `/var/MobileDeviceProvisioningProfiles` ed è possibile controllarli con **`security cms -D -i /path/to/profile`**

## **libmis.dyld**

Questa è la libreria esterna che `amfid` chiama per chiedere se dovrebbe consentire qualcosa o meno. Questo è stato storicamente abusato nel jailbreak eseguendo una versione backdoored di essa che consentirebbe tutto.

In macOS questo si trova all'interno di `MobileDevice.framework`.

## Cache di Fiducia AMFI

iOS AMFI mantiene un elenco di hash noti che sono firmati ad-hoc, chiamato **Trust Cache** e trovato nella sezione `__TEXT.__const` del kext. Nota che in operazioni molto specifiche e sensibili è possibile estendere questa Trust Cache con un file esterno.

## Riferimenti

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
