# macOS Authorizations DB & Authd



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

## **Athorizarions DB**

Baza podataka smeštena u `/var/db/auth.db` je baza podataka koja se koristi za čuvanje dozvola za obavljanje osetljivih operacija. Ove operacije se obavljaju potpuno u **korisničkom prostoru** i obično ih koriste **XPC servisi** koji treba da provere **da li je pozivajući klijent ovlašćen** da izvrši određenu radnju proverom ove baze podataka.

U početku, ova baza podataka se kreira iz sadržaja `/System/Library/Security/authorization.plist`. Zatim, neki servisi mogu dodati ili izmeniti ovu bazu podataka kako bi dodali druge dozvole.

Pravila se čuvaju u `rules` tabeli unutar baze podataka i sadrže sledeće kolone:

* **id**: Jedinstveni identifikator za svako pravilo, automatski se povećava i služi kao primarni ključ.
* **name**: Jedinstveno ime pravila koje se koristi za identifikaciju i referenciranje unutar sistema autorizacije.
* **type**: Određuje tip pravila, ograničeno na vrednosti 1 ili 2 za definisanje njegove logike autorizacije.
* **class**: Kategorizuje pravilo u određenu klasu, osiguravajući da je pozitivni ceo broj.
* "allow" za dozvolu, "deny" za odbijanje, "user" ako grupna svojstva ukazuju na grupu članstvo koje omogućava pristup, "rule" ukazuje u nizu na pravilo koje treba ispuniti, "evaluate-mechanisms" praćeno nizom `mechanisms` koji su ili ugrađeni ili naziv paketa unutar `/System/Library/CoreServices/SecurityAgentPlugins/` ili /Library/Security//SecurityAgentPlugins
* **group**: Ukazuje na korisničku grupu povezanu sa pravilom za autorizaciju zasnovanu na grupi.
* **kofn**: Predstavlja "k-of-n" parametar, određujući koliko podpravila mora biti ispunjeno od ukupnog broja.
* **timeout**: Definiše trajanje u sekundama pre nego što autorizacija koju dodeljuje pravilo istekne.
* **flags**: Sadrži razne oznake koje modifikuju ponašanje i karakteristike pravila.
* **tries**: Ograničava broj dozvoljenih pokušaja autorizacije radi poboljšanja bezbednosti.
* **version**: Prati verziju pravila za kontrolu verzija i ažuriranja.
* **created**: Beleži vremensku oznaku kada je pravilo kreirano u svrhe revizije.
* **modified**: Čuva vremensku oznaku poslednje izmene izvršene na pravilu.
* **hash**: Drži hash vrednost pravila kako bi se osigurala njegova celovitost i otkrila manipulacija.
* **identifier**: Pruža jedinstveni string identifikator, kao što je UUID, za spoljne reference na pravilo.
* **requirement**: Sadrži serijalizovane podatke koji definišu specifične zahteve i mehanizme autorizacije pravila.
* **comment**: Nudi opis ili komentar o pravilu koji je razumljiv ljudima za dokumentaciju i jasnoću.

### Example
```bash
# List by name and comments
sudo sqlite3 /var/db/auth.db "select name, comment from rules"

# Get rules for com.apple.tcc.util.admin
security authorizationdb read com.apple.tcc.util.admin
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>class</key>
<string>rule</string>
<key>comment</key>
<string>For modification of TCC settings.</string>
<key>created</key>
<real>701369782.01043606</real>
<key>modified</key>
<real>701369782.01043606</real>
<key>rule</key>
<array>
<string>authenticate-admin-nonshared</string>
</array>
<key>version</key>
<integer>0</integer>
</dict>
</plist>
```
Osim toga, na [https://www.dssw.co.uk/reference/authorization-rights/authenticate-admin-nonshared/](https://www.dssw.co.uk/reference/authorization-rights/authenticate-admin-nonshared/) moguće je videti značenje `authenticate-admin-nonshared`:
```json
{
'allow-root' : 'false',
'authenticate-user' : 'true',
'class' : 'user',
'comment' : 'Authenticate as an administrator.',
'group' : 'admin',
'session-owner' : 'false',
'shared' : 'false',
'timeout' : '30',
'tries' : '10000',
'version' : '1'
}
```
## Authd

To je demon koji prima zahteve za autorizaciju klijenata da izvrše osetljive radnje. Radi kao XPC servis definisan unutar `XPCServices/` foldera i koristi da piše svoje logove u `/var/log/authd.log`.

Pored toga, korišćenjem bezbednosnog alata moguće je testirati mnoge `Security.framework` API-je. Na primer, `AuthorizationExecuteWithPrivileges` se pokreće: `security execute-with-privileges /bin/ls`

To će fork-ovati i izvršiti `/usr/libexec/security_authtrampoline /bin/ls` kao root, što će tražiti dozvole u promptu da izvrši ls kao root:

<figure><img src="../../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

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
