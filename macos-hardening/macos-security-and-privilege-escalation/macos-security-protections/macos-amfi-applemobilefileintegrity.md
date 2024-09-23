# macOS - AMFI - AppleMobileFileIntegrity

{% hint style="success" %}
Lernen & üben Sie AWS Hacking:<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
Lernen & üben Sie GCP Hacking: <img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Unterstützen Sie HackTricks</summary>

* Überprüfen Sie die [**Abonnementpläne**](https://github.com/sponsors/carlospolop)!
* **Treten Sie der** 💬 [**Discord-Gruppe**](https://discord.gg/hRep4RUj7f) oder der [**Telegram-Gruppe**](https://t.me/peass) bei oder **folgen** Sie uns auf **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Teilen Sie Hacking-Tricks, indem Sie PRs an die** [**HackTricks**](https://github.com/carlospolop/hacktricks) und [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) GitHub-Repos senden.

</details>
{% endhint %}



## AppleMobileFileIntegrity.kext und amfid

Es konzentriert sich darauf, die Integrität des auf dem System ausgeführten Codes durch die Logik hinter der Code-Signaturüberprüfung von XNU durchzusetzen. Es kann auch Berechtigungen überprüfen und andere sensible Aufgaben wie das Erlauben von Debugging oder das Erhalten von Task-Ports übernehmen.

Darüber hinaus zieht es für einige Operationen vor, den im Benutzerspeicher ausgeführten Daemon `/usr/libexec/amfid` zu kontaktieren. Diese Vertrauensbeziehung wurde in mehreren Jailbreaks missbraucht.

AMFI verwendet **MACF**-Richtlinien und registriert seine Hooks, sobald es gestartet wird. Auch das Verhindern des Ladens oder Entladens könnte einen Kernel-Panik auslösen. Es gibt jedoch einige Boot-Argumente, die es ermöglichen, AMFI zu schwächen:

* `amfi_unrestricted_task_for_pid`: Erlaubt task\_for\_pid ohne erforderliche Berechtigungen
* `amfi_allow_any_signature`: Erlaubt jede Code-Signatur
* `cs_enforcement_disable`: Systemweites Argument zum Deaktivieren der Durchsetzung der Code-Signatur
* `amfi_prevent_old_entitled_platform_binaries`: Ungültige Plattform-Binärdateien mit Berechtigungen
* `amfi_get_out_of_my_way`: Deaktiviert amfi vollständig

Dies sind einige der MACF-Richtlinien, die es registriert:

* **`cred_check_label_update_execve:`** Label-Update wird durchgeführt und gibt 1 zurück
* **`cred_label_associate`**: Aktualisiert AMFIs mac-Label-Slot mit Label
* **`cred_label_destroy`**: Entfernt AMFIs mac-Label-Slot
* **`cred_label_init`**: Setzt 0 in AMFIs mac-Label-Slot
* **`cred_label_update_execve`:** Überprüft die Berechtigungen des Prozesses, um zu sehen, ob er die Labels ändern darf.
* **`file_check_mmap`:** Überprüft, ob mmap Speicher anfordert und ihn als ausführbar festlegt. In diesem Fall wird überprüft, ob eine Bibliotheksvalidierung erforderlich ist, und falls ja, wird die Funktion zur Bibliotheksvalidierung aufgerufen.
* **`file_check_library_validation`**: Ruft die Funktion zur Bibliotheksvalidierung auf, die unter anderem überprüft, ob eine Plattform-Binärdatei eine andere Plattform-Binärdatei lädt oder ob der Prozess und die neu geladene Datei die gleiche Team-ID haben. Bestimmte Berechtigungen erlauben auch das Laden beliebiger Bibliotheken.
* **`policy_initbsd`**: Richtet vertrauenswürdige NVRAM-Schlüssel ein
* **`policy_syscall`**: Überprüft DYLD-Richtlinien, wie ob die Binärdatei uneingeschränkte Segmente hat, ob Umgebungsvariablen erlaubt werden sollten... dies wird auch aufgerufen, wenn ein Prozess über `amfi_check_dyld_policy_self()` gestartet wird.
* **`proc_check_inherit_ipc_ports`**: Überprüft, ob, wenn ein Prozess eine neue Binärdatei ausführt, andere Prozesse mit SEND-Rechten über den Task-Port des Prozesses diese behalten sollten oder nicht. Plattform-Binärdateien sind erlaubt, `get-task-allow` berechtigt dazu, `task_for_pid-allow` Berechtigungen sind erlaubt und Binärdateien mit der gleichen Team-ID.
* **`proc_check_expose_task`**: Durchsetzt Berechtigungen
* **`amfi_exc_action_check_exception_send`**: Eine Ausnahme-Nachricht wird an den Debugger gesendet
* **`amfi_exc_action_label_associate & amfi_exc_action_label_copy/populate & amfi_exc_action_label_destroy & amfi_exc_action_label_init & amfi_exc_action_label_update`**: Label-Lebenszyklus während der Ausnahmebehandlung (Debugging)
* **`proc_check_get_task`**: Überprüft Berechtigungen wie `get-task-allow`, die anderen Prozessen erlauben, den Task-Port zu erhalten, und `task_for_pid-allow`, die es dem Prozess erlauben, die Task-Ports anderer Prozesse zu erhalten. Wenn keiner von beiden, wird `amfid permitunrestricteddebugging` aufgerufen, um zu überprüfen, ob es erlaubt ist.
* **`proc_check_mprotect`**: Verweigert, wenn `mprotect` mit dem Flag `VM_PROT_TRUSTED` aufgerufen wird, was darauf hinweist, dass der Bereich so behandelt werden muss, als ob er eine gültige Code-Signatur hat.
* **`vnode_check_exec`**: Wird aufgerufen, wenn ausführbare Dateien in den Speicher geladen werden und setzt `cs_hard | cs_kill`, was den Prozess tötet, wenn eine der Seiten ungültig wird
* **`vnode_check_getextattr`**: MacOS: Überprüft `com.apple.root.installed` und `isVnodeQuarantined()`
* **`vnode_check_setextattr`**: Wie get + com.apple.private.allow-bless und interne Installer-äquivalente Berechtigung
* &#x20;**`vnode_check_signature`**: Code, der XNU aufruft, um die Code-Signatur unter Verwendung von Berechtigungen, Vertrauenscache und `amfid` zu überprüfen
* &#x20;**`proc_check_run_cs_invalid`**: Es interceptiert `ptrace()`-Aufrufe (`PT_ATTACH` und `PT_TRACE_ME`). Es überprüft auf eine der Berechtigungen `get-task-allow`, `run-invalid-allow` und `run-unsigned-code` und wenn keine, wird überprüft, ob Debugging erlaubt ist.
* **`proc_check_map_anon`**: Wenn mmap mit dem **`MAP_JIT`**-Flag aufgerufen wird, überprüft AMFI die Berechtigung `dynamic-codesigning`.

`AMFI.kext` bietet auch eine API für andere Kernel-Erweiterungen, und es ist möglich, seine Abhängigkeiten mit zu finden:
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

Dies ist der Benutzer-Modus-Daemon, den `AMFI.kext` verwendet, um Code-Signaturen im Benutzermodus zu überprüfen.\
Damit `AMFI.kext` mit dem Daemon kommunizieren kann, verwendet es Mach-Nachrichten über den Port `HOST_AMFID_PORT`, der der spezielle Port `18` ist.

Beachten Sie, dass es in macOS nicht mehr möglich ist, dass Root-Prozesse spezielle Ports übernehmen, da sie durch `SIP` geschützt sind und nur launchd sie erhalten kann. In iOS wird überprüft, ob der Prozess, der die Antwort zurücksendet, den hardcodierten CDHash von `amfid` hat.

Es ist möglich zu sehen, wann `amfid` angefordert wird, um eine Binärdatei zu überprüfen, und die Antwort darauf, indem man es debuggt und einen Haltepunkt in `mach_msg` setzt.

Sobald eine Nachricht über den speziellen Port empfangen wird, wird **MIG** verwendet, um jede Funktion an die Funktion zu senden, die sie aufruft. Die Hauptfunktionen wurden umgekehrt und im Buch erklärt.

## Provisioning Profiles

Ein Provisioning-Profil kann verwendet werden, um Code zu signieren. Es gibt **Developer**-Profile, die verwendet werden können, um Code zu signieren und zu testen, und **Enterprise**-Profile, die auf allen Geräten verwendet werden können.

Nachdem eine App im Apple Store eingereicht wurde, wird sie, wenn sie genehmigt wird, von Apple signiert und das Provisioning-Profil wird nicht mehr benötigt.

Ein Profil verwendet normalerweise die Erweiterung `.mobileprovision` oder `.provisionprofile` und kann mit folgendem Befehl ausgegeben werden:
```bash
openssl asn1parse -inform der -in /path/to/profile

# Or

security cms -D -i /path/to/profile
```
Obwohl manchmal als zertifiziert bezeichnet, haben diese Bereitstellungsprofile mehr als ein Zertifikat:

* **AppIDName:** Die Anwendungskennung
* **AppleInternalProfile**: Bezeichnet dies als ein internes Apple-Profil
* **ApplicationIdentifierPrefix**: Vorangestellt an AppIDName (gleich wie TeamIdentifier)
* **CreationDate**: Datum im Format `YYYY-MM-DDTHH:mm:ssZ`
* **DeveloperCertificates**: Ein Array von (normalerweise einem) Zertifikat(en), kodiert als Base64-Daten
* **Entitlements**: Die Berechtigungen, die mit Berechtigungen für dieses Profil erlaubt sind
* **ExpirationDate**: Ablaufdatum im Format `YYYY-MM-DDTHH:mm:ssZ`
* **Name**: Der Anwendungsname, derselbe wie AppIDName
* **ProvisionedDevices**: Ein Array (für Entwicklerzertifikate) von UDIDs, für die dieses Profil gültig ist
* **ProvisionsAllDevices**: Ein Boolean (true für Unternehmenszertifikate)
* **TeamIdentifier**: Ein Array von (normalerweise einem) alphanumerischen Zeichenfolge(n), die verwendet wird, um den Entwickler für inter-app Interaktionszwecke zu identifizieren
* **TeamName**: Ein menschenlesbarer Name, der verwendet wird, um den Entwickler zu identifizieren
* **TimeToLive**: Gültigkeit (in Tagen) des Zertifikats
* **UUID**: Eine universell eindeutige Kennung für dieses Profil
* **Version**: Derzeit auf 1 gesetzt

Beachten Sie, dass der Eintrag für Berechtigungen eine eingeschränkte Menge an Berechtigungen enthalten wird und das Bereitstellungsprofil nur diese spezifischen Berechtigungen vergeben kann, um zu verhindern, dass Apple private Berechtigungen gewährt.

Beachten Sie, dass Profile normalerweise in `/var/MobileDeviceProvisioningProfiles` gespeichert sind und es möglich ist, sie mit **`security cms -D -i /path/to/profile`** zu überprüfen.

## **libmis.dyld**

Dies ist die externe Bibliothek, die `amfid` aufruft, um zu fragen, ob es etwas erlauben soll oder nicht. Dies wurde historisch beim Jailbreaking missbraucht, indem eine gehackte Version davon ausgeführt wurde, die alles erlaubte.

In macOS befindet sich dies innerhalb von `MobileDevice.framework`.

## AMFI Trust Caches

iOS AMFI verwaltet eine Liste bekannter Hashes, die ad-hoc signiert sind, genannt **Trust Cache** und im kext's `__TEXT.__const` Abschnitt gefunden werden. Beachten Sie, dass es bei sehr spezifischen und sensiblen Operationen möglich ist, diesen Trust Cache mit einer externen Datei zu erweitern.

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
