# macOS Gatekeeper / Quarantine / XProtect

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

<figure><img src="https://pentest.eu/RENDER_WebSec_10fps_21sec_9MB_29042024.gif" alt=""><figcaption></figcaption></figure>

{% embed url="https://websec.nl/" %}

## Gatekeeper

**Gatekeeper**は、Macオペレーティングシステム向けに開発されたセキュリティ機能で、ユーザーが**信頼できるソフトウェアのみを実行する**ことを保証するように設計されています。これは、ユーザーが**App Store以外のソース**からダウンロードして開こうとするソフトウェア（アプリ、プラグイン、インストーラーパッケージなど）を**検証する**ことによって機能します。

Gatekeeperの主要なメカニズムは、その**検証**プロセスにあります。ダウンロードされたソフトウェアが**認識された開発者によって署名されているか**を確認し、ソフトウェアの真正性を保証します。さらに、ソフトウェアが**Appleによって公証されているか**を確認し、既知の悪意のあるコンテンツが含まれておらず、公証後に改ざんされていないことを確認します。

加えて、Gatekeeperは、**ユーザーにダウンロードしたソフトウェアを初めて開くことを承認するよう促す**ことで、ユーザーの制御とセキュリティを強化します。この保護機能は、ユーザーが無害なデータファイルと誤解して実行してしまう可能性のある有害な実行可能コードを誤って実行するのを防ぐのに役立ちます。

### Application Signatures

アプリケーション署名、またはコード署名は、Appleのセキュリティインフラストラクチャの重要な要素です。これらは、**ソフトウェアの著者（開発者）の身元を確認する**ために使用され、コードが最後に署名されて以来改ざんされていないことを保証します。

以下はその仕組みです：

1. **アプリケーションの署名：** 開発者がアプリケーションを配布する準備が整ったとき、彼らは**プライベートキーを使用してアプリケーションに署名します**。このプライベートキーは、Apple Developer Programに登録した際にAppleが開発者に発行する**証明書**に関連付けられています。署名プロセスは、アプリのすべての部分の暗号ハッシュを作成し、このハッシュを開発者のプライベートキーで暗号化することを含みます。
2. **アプリケーションの配布：** 署名されたアプリケーションは、開発者の証明書と共にユーザーに配布されます。この証明書には、対応する公開鍵が含まれています。
3. **アプリケーションの検証：** ユーザーがアプリケーションをダウンロードして実行しようとすると、彼らのMacオペレーティングシステムは開発者の証明書から公開鍵を使用してハッシュを復号化します。その後、アプリケーションの現在の状態に基づいてハッシュを再計算し、これを復号化されたハッシュと比較します。一致すれば、**アプリケーションは開発者によって署名されて以来変更されていない**ことを意味し、システムはアプリケーションの実行を許可します。

アプリケーション署名は、AppleのGatekeeper技術の重要な部分です。ユーザーが**インターネットからダウンロードしたアプリケーションを開こうとすると**、Gatekeeperはアプリケーション署名を検証します。Appleが知られた開発者に発行した証明書で署名されており、コードが改ざんされていない場合、Gatekeeperはアプリケーションの実行を許可します。そうでない場合、アプリケーションはブロックされ、ユーザーに警告されます。

macOS Catalina以降、**GatekeeperはアプリケーションがAppleによって公証されているかどうかも確認します**。これにより、セキュリティの追加層が加わります。公証プロセスは、アプリケーションに既知のセキュリティ問題や悪意のあるコードがないかをチェックし、これらのチェックに合格すると、AppleはGatekeeperが検証できるチケットをアプリケーションに追加します。

#### Check Signatures

いくつかの**マルウェアサンプル**をチェックする際は、**署名を確認する**ことが常に重要です。署名した**開発者**がすでに**マルウェアに関連している**可能性があるためです。
```bash
# Get signer
codesign -vv -d /bin/ls 2>&1 | grep -E "Authority|TeamIdentifier"

# Check if the app’s contents have been modified
codesign --verify --verbose /Applications/Safari.app

# Get entitlements from the binary
codesign -d --entitlements :- /System/Applications/Automator.app # Check the TCC perms

# Check if the signature is valid
spctl --assess --verbose /Applications/Safari.app

# Sign a binary
codesign -s <cert-name-keychain> toolsdemo
```
### Notarization

Appleのノータリゼーションプロセスは、ユーザーを潜在的に有害なソフトウェアから保護するための追加の安全策として機能します。これは、**開発者が自分のアプリケーションを** **Appleのノータリーサービス**に審査のために提出することを含み、App Reviewとは混同しないでください。このサービスは、**自動化されたシステム**であり、提出されたソフトウェアに**悪意のあるコンテンツ**やコード署名に関する潜在的な問題がないかを精査します。

ソフトウェアがこの検査を問題なく通過すると、ノータリーサービスはノータリゼーションチケットを生成します。開発者はこのチケットを**自分のソフトウェアに添付する**必要があり、このプロセスは「ステープリング」として知られています。さらに、ノータリゼーションチケットはオンラインでも公開され、Appleのセキュリティ技術であるGatekeeperがアクセスできるようになります。

ユーザーがソフトウェアを初めてインストールまたは実行する際、ノータリゼーションチケットの存在 - 実行可能ファイルにステープルされているか、オンラインで見つかるかにかかわらず - **Gatekeeperに対してそのソフトウェアがAppleによってノータリゼーションされたことを通知します**。その結果、Gatekeeperは初回起動ダイアログに説明的なメッセージを表示し、そのソフトウェアがAppleによって悪意のあるコンテンツのチェックを受けたことを示します。このプロセスにより、ユーザーは自分のシステムにインストールまたは実行するソフトウェアのセキュリティに対する信頼が高まります。

### spctl & syspolicyd

{% hint style="danger" %}
Sequoiaバージョンから、**`spctl`**はもはやGatekeeperの設定を変更することを許可しませんので注意してください。
{% endhint %}

**`spctl`**は、Gatekeeperと対話し、列挙するためのCLIツールです（XPCメッセージを介して`syspolicyd`デーモンと）。例えば、次のコマンドでGatekeeperの**ステータス**を確認することができます:
```bash
# Check the status
spctl --status
```
{% hint style="danger" %}
GateKeeperの署名チェックは、**Quarantine属性を持つファイル**に対してのみ実行され、すべてのファイルに対して行われるわけではありません。
{% endhint %}

GateKeeperは、**設定と署名**に基づいてバイナリが実行可能かどうかを確認します：

<figure><img src="../../../.gitbook/assets/image (1150).png" alt=""><figcaption></figcaption></figure>

**`syspolicyd`**は、Gatekeeperを強制するための主要なデーモンです。これは`/var/db/SystemPolicy`にあるデータベースを維持しており、[データベースをサポートするコードはこちら](https://opensource.apple.com/source/Security/Security-58286.240.4/OSX/libsecurity\_codesigning/lib/policydb.cpp)と[SQLテンプレートはこちら](https://opensource.apple.com/source/Security/Security-58286.240.4/OSX/libsecurity\_codesigning/lib/syspolicy.sql)で見つけることができます。データベースはSIPによって制限されておらず、rootによって書き込み可能で、データベース`/var/db/.SystemPolicy-default`は、他のデータベースが破損した場合の元のバックアップとして使用されます。

さらに、バンドル**`/var/db/gke.bundle`**と**`/var/db/gkopaque.bundle`**には、データベースに挿入されるルールを含むファイルがあります。このデータベースはrootとして次のコマンドで確認できます：
```bash
# Open database
sqlite3 /var/db/SystemPolicy

# Get allowed rules
SELECT requirement,allow,disabled,label from authority where label != 'GKE' and disabled=0;
requirement|allow|disabled|label
anchor apple generic and certificate 1[subject.CN] = "Apple Software Update Certification Authority"|1|0|Apple Installer
anchor apple|1|0|Apple System
anchor apple generic and certificate leaf[field.1.2.840.113635.100.6.1.9] exists|1|0|Mac App Store
anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and (certificate leaf[field.1.2.840.113635.100.6.1.14] or certificate leaf[field.1.2.840.113635.100.6.1.13]) and notarized|1|0|Notarized Developer ID
[...]
```
**`syspolicyd`** は、`assess`、`update`、`record`、`cancel` などの異なる操作を持つ XPC サーバーも公開しており、これらは **`Security.framework`** の `SecAssessment*` API を使用してアクセス可能です。また、**`xpctl`** は実際に XPC を介して **`syspolicyd`** と通信します。

最初のルールが "**App Store**" で終わり、2 番目のルールが "**Developer ID**" で終わることに注意してください。また、前の画像では **App Store と認識された開発者からのアプリを実行することが有効** でした。\
その設定を App Store に **変更**すると、"**Notarized Developer ID" ルールは消えます**。

**type GKE** のルールも数千あります：
```bash
SELECT requirement,allow,disabled,label from authority where label = 'GKE' limit 5;
cdhash H"b40281d347dc574ae0850682f0fd1173aa2d0a39"|1|0|GKE
cdhash H"5fd63f5342ac0c7c0774ebcbecaf8787367c480f"|1|0|GKE
cdhash H"4317047eefac8125ce4d44cab0eb7b1dff29d19a"|1|0|GKE
cdhash H"0a71962e7a32f0c2b41ddb1fb8403f3420e1d861"|1|0|GKE
cdhash H"8d0d90ff23c3071211646c4c9c607cdb601cb18f"|1|0|GKE
```
これらは次の場所からのハッシュです：

* `/var/db/SystemPolicyConfiguration/gke.bundle/Contents/Resources/gke.auth`
* `/var/db/gke.bundle/Contents/Resources/gk.db`
* `/var/db/gkopaque.bundle/Contents/Resources/gkopaque.db`

または、次のコマンドで前の情報をリストすることができます：
```bash
sudo spctl --list
```
オプション **`--master-disable`** と **`--global-disable`** の **`spctl`** は、これらの署名チェックを完全に **無効に** します：
```bash
# Disable GateKeeper
spctl --global-disable
spctl --master-disable

# Enable it
spctl --global-enable
spctl --master-enable
```
完全に有効にすると、新しいオプションが表示されます：

<figure><img src="../../../.gitbook/assets/image (1151).png" alt=""><figcaption></figcaption></figure>

**アプリがGateKeeperによって許可されるかどうかを確認することができます**：
```bash
spctl --assess -v /Applications/App.app
```
GateKeeperに新しいルールを追加して、特定のアプリの実行を許可することが可能です：
```bash
# Check if allowed - nop
spctl --assess -v /Applications/App.app
/Applications/App.app: rejected
source=no usable signature

# Add a label and allow this label in GateKeeper
sudo spctl --add --label "whitelist" /Applications/App.app
sudo spctl --enable --label "whitelist"

# Check again - yep
spctl --assess -v /Applications/App.app
/Applications/App.app: accepted
```
Regarding **kernel extensions**, the folder `/var/db/SystemPolicyConfiguration` contains files with lists of kexts allowed to be loaded. Moreover, `spctl` has the entitlement `com.apple.private.iokit.nvram-csr` because it's capable of  adding new pre-approved kernel extensions which need to be saved also in NVRAM in a `kext-allowed-teams` key.

### Quarantine Files

Upon **downloading** an application or file, specific macOS **applications** such as web browsers or email clients **attach an extended file attribute**, commonly known as the "**quarantine flag**," to the downloaded file. This attribute acts as a security measure to **mark the file** as coming from an untrusted source (the internet), and potentially carrying risks. However, not all applications attach this attribute, for instance, common BitTorrent client software usually bypasses this process.

**The presence of a quarantine flag signals macOS's Gatekeeper security feature when a user attempts to execute the file**.

In the case where the **quarantine flag is not present** (as with files downloaded via some BitTorrent clients), Gatekeeper's **checks may not be performed**. Thus, users should exercise caution when opening files downloaded from less secure or unknown sources.

{% hint style="info" %}
**コード署名の** **有効性**を**確認する**ことは、コードとそのバンドルされたリソースの暗号学的**ハッシュ**を生成することを含む**リソース集約的**なプロセスです。さらに、証明書の有効性を確認するには、発行後に取り消されたかどうかを確認するためにAppleのサーバーに**オンラインチェック**を行う必要があります。これらの理由から、アプリが起動するたびに完全なコード署名とノータリゼーションのチェックを実行することは**非現実的です**。

したがって、これらのチェックは**隔離属性を持つアプリを実行する際にのみ実行されます。**
{% endhint %}

{% hint style="warning" %}
この属性は**ファイルを作成/ダウンロードするアプリケーションによって設定される必要があります**。

ただし、サンドボックス化されたファイルは、作成するすべてのファイルにこの属性が設定されます。サンドボックス化されていないアプリは、自分で設定するか、**Info.plist**に[**LSFileQuarantineEnabled**](https://developer.apple.com/documentation/bundleresources/information_property_list/lsfilequarantineenabled?language=objc)キーを指定することで、作成されたファイルに`com.apple.quarantine`拡張属性を設定させることができます。
{% endhint %}

Moreover, all files created by a process calling **`qtn_proc_apply_to_self`** are quarantined. Or the API **`qtn_file_apply_to_path`** adds the quarantine attribute to a specified file path.

It's possible to **check it's status and enable/disable** (root required) with:
```bash
spctl --status
assessments enabled

spctl --enable
spctl --disable
#You can also allow nee identifies to execute code using the binary "spctl"
```
ファイルが隔離拡張属性を持っているかどうかを**確認することもできます**:
```bash
xattr file.png
com.apple.macl
com.apple.quarantine
```
**拡張属性**の**値**を確認し、次のコマンドで隔離属性を書き込んだアプリを特定します:
```bash
xattr -l portada.png
com.apple.macl:
00000000  03 00 53 DA 55 1B AE 4C 4E 88 9D CA B7 5C 50 F3  |..S.U..LN.....P.|
00000010  16 94 03 00 27 63 64 97 98 FB 4F 02 84 F3 D0 DB  |....'cd...O.....|
00000020  89 53 C3 FC 03 00 27 63 64 97 98 FB 4F 02 84 F3  |.S....'cd...O...|
00000030  D0 DB 89 53 C3 FC 00 00 00 00 00 00 00 00 00 00  |...S............|
00000040  00 00 00 00 00 00 00 00                          |........|
00000048
com.apple.quarantine: 00C1;607842eb;Brave;F643CD5F-6071-46AB-83AB-390BA944DEC5
# 00c1 -- It has been allowed to eexcute this file (QTN_FLAG_USER_APPROVED = 0x0040)
# 607842eb -- Timestamp
# Brave -- App
# F643CD5F-6071-46AB-83AB-390BA944DEC5 -- UID assigned to the file downloaded
```
実際、プロセスは「作成したファイルに対して隔離フラグを設定できる」（作成したファイルにUSER_APPROVEDフラグを適用しようとしましたが、適用されませんでした）：

<details>

<summary>ソースコード 隔離フラグを適用</summary>
```c
#include <stdio.h>
#include <stdlib.h>

enum qtn_flags {
QTN_FLAG_DOWNLOAD = 0x0001,
QTN_FLAG_SANDBOX = 0x0002,
QTN_FLAG_HARD = 0x0004,
QTN_FLAG_USER_APPROVED = 0x0040,
};

#define qtn_proc_alloc _qtn_proc_alloc
#define qtn_proc_apply_to_self _qtn_proc_apply_to_self
#define qtn_proc_free _qtn_proc_free
#define qtn_proc_init _qtn_proc_init
#define qtn_proc_init_with_self _qtn_proc_init_with_self
#define qtn_proc_set_flags _qtn_proc_set_flags
#define qtn_file_alloc _qtn_file_alloc
#define qtn_file_init_with_path _qtn_file_init_with_path
#define qtn_file_free _qtn_file_free
#define qtn_file_apply_to_path _qtn_file_apply_to_path
#define qtn_file_set_flags _qtn_file_set_flags
#define qtn_file_get_flags _qtn_file_get_flags
#define qtn_proc_set_identifier _qtn_proc_set_identifier

typedef struct _qtn_proc *qtn_proc_t;
typedef struct _qtn_file *qtn_file_t;

int qtn_proc_apply_to_self(qtn_proc_t);
void qtn_proc_init(qtn_proc_t);
int qtn_proc_init_with_self(qtn_proc_t);
int qtn_proc_set_flags(qtn_proc_t, uint32_t flags);
qtn_proc_t qtn_proc_alloc();
void qtn_proc_free(qtn_proc_t);
qtn_file_t qtn_file_alloc(void);
void qtn_file_free(qtn_file_t qf);
int qtn_file_set_flags(qtn_file_t qf, uint32_t flags);
uint32_t qtn_file_get_flags(qtn_file_t qf);
int qtn_file_apply_to_path(qtn_file_t qf, const char *path);
int qtn_file_init_with_path(qtn_file_t qf, const char *path);
int qtn_proc_set_identifier(qtn_proc_t qp, const char* bundleid);

int main() {

qtn_proc_t qp = qtn_proc_alloc();
qtn_proc_set_identifier(qp, "xyz.hacktricks.qa");
qtn_proc_set_flags(qp, QTN_FLAG_DOWNLOAD | QTN_FLAG_USER_APPROVED);
qtn_proc_apply_to_self(qp);
qtn_proc_free(qp);

FILE *fp;
fp = fopen("thisisquarantined.txt", "w+");
fprintf(fp, "Hello Quarantine\n");
fclose(fp);

return 0;

}
```
</details>

そして、その属性を**削除**するには:
```bash
xattr -d com.apple.quarantine portada.png
#You can also remove this attribute from every file with
find . -iname '*' -print0 | xargs -0 xattr -d com.apple.quarantine
```
そして、次のコマンドで隔離されたファイルをすべて見つけます：

{% code overflow="wrap" %}
```bash
find / -exec ls -ld {} \; 2>/dev/null | grep -E "[x\-]@ " | awk '{printf $9; printf "\n"}' | xargs -I {} xattr -lv {} | grep "com.apple.quarantine"
```
{% endcode %}

検疫情報は、**`~/Library/Preferences/com.apple.LaunchServices.QuarantineEventsV2`** にあるLaunchServicesによって管理される中央データベースに保存されており、GUIがファイルの起源に関するデータを取得できるようにします。さらに、これは起源を隠そうとするアプリケーションによって上書きされる可能性があります。さらに、これはLaunchServices APIから行うことができます。

#### **libquarantine.dylb**

このライブラリは、拡張属性フィールドを操作するためのいくつかの関数をエクスポートします。

`qtn_file_*` APIはファイル検疫ポリシーを扱い、`qtn_proc_*` APIはプロセス（プロセスによって作成されたファイル）に適用されます。エクスポートされていない`__qtn_syscall_quarantine*`関数は、ポリシーを適用するもので、最初の引数として「Quarantine」を指定して`mac_syscall`を呼び出し、`Quarantine.kext`にリクエストを送信します。

#### **Quarantine.kext**

カーネル拡張は、**システムのカーネルキャッシュを通じてのみ利用可能**ですが、**[**https://developer.apple.com/**](https://developer.apple.com/)から**カーネルデバッグキットをダウンロードすることができ、拡張のシンボリケートされたバージョンが含まれています。

このKextは、MACFを介していくつかの呼び出しをフックし、すべてのファイルライフサイクルイベントをトラップします：作成、オープン、名前変更、ハードリンク... さらには`setxattr`を使用して`com.apple.quarantine`拡張属性の設定を防ぎます。

いくつかのMIBも使用します：

* `security.mac.qtn.sandbox_enforce`: サンドボックスに沿った検疫を強制
* `security.mac.qtn.user_approved_exec`: 検疫されたプロセスは承認されたファイルのみを実行可能

### XProtect

XProtectはmacOSに組み込まれた**アンチマルウェア**機能です。XProtectは、アプリケーションが最初に起動または変更されたときに、既知のマルウェアおよび安全でないファイルタイプのデータベースに対して**チェックを行います**。Safari、Mail、Messagesなどの特定のアプリを通じてファイルをダウンロードすると、XProtectは自動的にファイルをスキャンします。ファイルがデータベース内の既知のマルウェアと一致する場合、XProtectは**ファイルの実行を防ぎ**、脅威を警告します。

XProtectデータベースは、Appleによって**定期的に更新**され、新しいマルウェア定義が追加されます。これらの更新は自動的にダウンロードされ、Macにインストールされます。これにより、XProtectは常に最新の既知の脅威に対して最新の状態を保ちます。

ただし、**XProtectはフル機能のアンチウイルスソリューションではない**ことに注意する価値があります。特定の既知の脅威のリストをチェックするだけであり、ほとんどのアンチウイルスソフトウェアのようにオンアクセススキャンを実行しません。

最新のXProtect更新に関する情報を取得するには、次のコマンドを実行します：

{% code overflow="wrap" %}
```bash
system_profiler SPInstallHistoryDataType 2>/dev/null | grep -A 4 "XProtectPlistConfigData" | tail -n 5
```
{% endcode %}

XProtectは、**/Library/Apple/System/Library/CoreServices/XProtect.bundle**のSIP保護された場所にあり、バンドル内にはXProtectが使用する情報が含まれています：

* **`XProtect.bundle/Contents/Resources/LegacyEntitlementAllowlist.plist`**: これらのcdhashを持つコードがレガシー権限を使用できるようにします。
* **`XProtect.bundle/Contents/Resources/XProtect.meta.plist`**: BundleIDおよびTeamIDを介して読み込むことが禁止されているプラグインと拡張機能のリスト、または最小バージョンを示します。
* **`XProtect.bundle/Contents/Resources/XProtect.yara`**: マルウェアを検出するためのYaraルール。
* **`XProtect.bundle/Contents/Resources/gk.db`**: ブロックされたアプリケーションとTeamIDのハッシュを含むSQLite3データベース。

**`/Library/Apple/System/Library/CoreServices/XProtect.app`**には、Gatekeeperプロセスに関与しないXProtectに関連する別のアプリがあります。

### Not Gatekeeper

{% hint style="danger" %}
Gatekeeperは、アプリケーションを実行するたびに**実行されるわけではありません**。実際には、_**AppleMobileFileIntegrity**_ (AMFI)は、Gatekeeperによってすでに実行され、検証されたアプリを実行する際にのみ**実行可能コードの署名を検証**します。
{% endhint %}

したがって、以前はアプリを実行してGatekeeperでキャッシュし、その後**アプリケーションの実行可能でないファイルを変更する**（ElectronのasarやNIBファイルなど）ことが可能でしたが、他の保護がなければ、アプリケーションは**悪意のある**追加とともに**実行されました**。

しかし、現在はmacOSがアプリケーションバンドル内のファイルの**変更を防ぐ**ため、これは不可能です。したがって、[Dirty NIB](../macos-proces-abuse/macos-dirty-nib.md)攻撃を試みると、Gatekeeperでキャッシュするためにアプリを実行した後、バンドルを変更できなくなるため、もはや悪用できないことがわかります。たとえば、Contentsディレクトリの名前をNotConに変更し（エクスプロイトで示されているように）、その後アプリのメインバイナリを実行してGatekeeperでキャッシュしようとすると、エラーが発生し、実行されません。

## Gatekeeper Bypasses

Gatekeeperをバイパスする方法（ユーザーに何かをダウンロードさせ、Gatekeeperがそれを拒否すべきときに実行させること）は、macOSの脆弱性と見なされます。以下は、過去にGatekeeperをバイパスすることを可能にした技術に割り当てられたCVEのいくつかです：

### [CVE-2021-1810](https://labs.withsecure.com/publications/the-discovery-of-cve-2021-1810)

**Archive Utility**を使用して抽出すると、**886文字を超えるパス**を持つファイルがcom.apple.quarantine拡張属性を受け取らないことが観察されました。この状況は、これらのファイルが**Gatekeeperの**セキュリティチェックを**回避する**ことを意図せずに許可します。

詳細については、[**元の報告**](https://labs.withsecure.com/publications/the-discovery-of-cve-2021-1810)を確認してください。

### [CVE-2021-30990](https://ronmasas.com/posts/bypass-macos-gatekeeper)

**Automator**で作成されたアプリケーションでは、実行に必要な情報が`application.app/Contents/document.wflow`内にあり、実行可能ファイルにはありません。実行可能ファイルは、**Automator Application Stub**と呼ばれる一般的なAutomatorバイナリです。

したがって、`application.app/Contents/MacOS/Automator\ Application\ Stub`を**シンボリックリンクでシステム内の別のAutomator Application Stubにポイントさせる**ことができ、`document.wflow`（あなたのスクリプト）内の内容を**Gatekeeperをトリガーせずに実行**します。

期待される場所の例：`/System/Library/CoreServices/Automator\ Application\ Stub.app/Contents/MacOS/Automator\ Application\ Stub`

詳細については、[**元の報告**](https://ronmasas.com/posts/bypass-macos-gatekeeper)を確認してください。

### [CVE-2022-22616](https://www.jamf.com/blog/jamf-threat-labs-safari-vuln-gatekeeper-bypass/)

このバイパスでは、`application.app`ではなく`application.app/Contents`から圧縮を開始するアプリケーションを含むzipファイルが作成されました。したがって、**quarantine attr**はすべての**`application.app/Contents`のファイル**に適用されましたが、**`application.app`には適用されませんでした**。これがGatekeeperがチェックしていたもので、`application.app`がトリガーされたときに**quarantine属性を持っていなかったため、Gatekeeperはバイパスされました。**
```bash
zip -r test.app/Contents test.zip
```
Check the [**original report**](https://www.jamf.com/blog/jamf-threat-labs-safari-vuln-gatekeeper-bypass/) for more information.

### [CVE-2022-32910](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-32910)

コンポーネントが異なっていても、この脆弱性の悪用は前のものと非常に似ています。この場合、**`application.app/Contents`** から Apple Archive を生成するため、**`application.app`** は **Archive Utility** によって解凍されるときに隔離属性を取得しません。
```bash
aa archive -d test.app/Contents -o test.app.aar
```
Check the [**original report**](https://www.jamf.com/blog/jamf-threat-labs-macos-archive-utility-vulnerability/) for more information.

### [CVE-2022-42821](https://www.microsoft.com/en-us/security/blog/2022/12/19/gatekeepers-achilles-heel-unearthing-a-macos-vulnerability/)

ACL **`writeextattr`** は、誰もファイルに属性を書き込むことを防ぐために使用できます：
```bash
touch /tmp/no-attr
chmod +a "everyone deny writeextattr" /tmp/no-attr
xattr -w attrname vale /tmp/no-attr
xattr: [Errno 13] Permission denied: '/tmp/no-attr'
```
さらに、**AppleDouble**ファイル形式は、ファイルをそのACEを含めてコピーします。

[**ソースコード**](https://opensource.apple.com/source/Libc/Libc-391/darwin/copyfile.c.auto.html)では、xattr内に保存されたACLのテキスト表現が**`com.apple.acl.text`**という名前で、解凍されたファイルのACLとして設定されることがわかります。したがって、ACLが他のxattrsの書き込みを防ぐように設定されたアプリケーションを**AppleDouble**ファイル形式でzipファイルに圧縮した場合... 検疫xattrはアプリケーションに設定されませんでした：

{% code overflow="wrap" %}
```bash
chmod +a "everyone deny write,writeattr,writeextattr" /tmp/test
ditto -c -k test test.zip
python3 -m http.server
# Download the zip from the browser and decompress it, the file should be without a quarantine xattr
```
{% endcode %}

詳細については、[**元のレポート**](https://www.microsoft.com/en-us/security/blog/2022/12/19/gatekeepers-achilles-heel-unearthing-a-macos-vulnerability/)を確認してください。

これはAppleArchivesを使用しても悪用される可能性があることに注意してください:
```bash
mkdir app
touch app/test
chmod +a "everyone deny write,writeattr,writeextattr" app/test
aa archive -d app -o test.aar
```
### [CVE-2023-27943](https://blog.f-secure.com/discovery-of-gatekeeper-bypass-cve-2023-27943/)

**Google Chromeがダウンロードしたファイルに対して隔離属性を設定していなかった**ことが、macOSの内部問題によって発見されました。

### [CVE-2023-27951](https://redcanary.com/blog/gatekeeper-bypass-vulnerabilities/)

AppleDoubleファイル形式は、ファイルの属性を`._`で始まる別のファイルに保存します。これにより、**macOSマシン間でファイル属性をコピーする**ことができます。しかし、AppleDoubleファイルを解凍した後、`._`で始まるファイルに**隔離属性が与えられなかった**ことが確認されました。

{% code overflow="wrap" %}
```bash
mkdir test
echo a > test/a
echo b > test/b
echo ._a > test/._a
aa archive -d test/ -o test.aar

# If you downloaded the resulting test.aar and decompress it, the file test/._a won't have a quarantitne attribute
```
{% endcode %}

クアランティン属性が設定されないファイルを作成できることで、**Gatekeeperをバイパスすることが可能でした。** トリックは、AppleDouble名付け規則を使用して**DMGファイルアプリケーションを作成し**（`._`で始める）、**この隠しファイルへのシンボリックリンクとして可視ファイルを作成することでした。**\
**dmgファイルが実行されると、**クアランティン属性がないため、**Gatekeeperをバイパスします。**
```bash
# Create an app bundle with the backdoor an call it app.app

echo "[+] creating disk image with app"
hdiutil create -srcfolder app.app app.dmg

echo "[+] creating directory and files"
mkdir
mkdir -p s/app
cp app.dmg s/app/._app.dmg
ln -s ._app.dmg s/app/app.dmg

echo "[+] compressing files"
aa archive -d s/ -o app.aar
```
### uchg (from this [talk](https://codeblue.jp/2023/result/pdf/cb23-bypassing-macos-security-and-privacy-mechanisms-from-gatekeeper-to-system-integrity-protection-by-koh-nakagawa.pdf))

* アプリを含むディレクトリを作成します。
* アプリにuchgを追加します。
* アプリをtar.gzファイルに圧縮します。
* tar.gzファイルを被害者に送信します。
* 被害者はtar.gzファイルを開き、アプリを実行します。
* Gatekeeperはアプリをチェックしません。

### Quarantine xattrを防ぐ

".app"バンドルにquarantine xattrが追加されていない場合、実行時に**Gatekeeperはトリガーされません**。

<figure><img src="https://pentest.eu/RENDER_WebSec_10fps_21sec_9MB_29042024.gif" alt=""><figcaption></figcaption></figure>

{% embed url="https://websec.nl/" %}

{% hint style="success" %}
AWSハッキングを学び、実践する：<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">[**HackTricks Training AWS Red Team Expert (ARTE)**](https://training.hacktricks.xyz/courses/arte)<img src="../../../.gitbook/assets/arte.png" alt="" data-size="line">\
GCPハッキングを学び、実践する：<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">[**HackTricks Training GCP Red Team Expert (GRTE)**<img src="../../../.gitbook/assets/grte.png" alt="" data-size="line">](https://training.hacktricks.xyz/courses/grte)

<details>

<summary>HackTricksをサポートする</summary>

* [**サブスクリプションプラン**](https://github.com/sponsors/carlospolop)を確認してください！
* **💬 [**Discordグループ**](https://discord.gg/hRep4RUj7f)または[**テレグラムグループ**](https://t.me/peass)に参加するか、**Twitter** 🐦 [**@hacktricks\_live**](https://twitter.com/hacktricks\_live)**をフォローしてください。**
* **ハッキングのトリックを共有するには、[**HackTricks**](https://github.com/carlospolop/hacktricks)および[**HackTricks Cloud**](https://github.com/carlospolop/hacktricks-cloud)のgithubリポジトリにPRを送信してください。**

</details>
{% endhint %}
