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



## AppleMobileFileIntegrity.kext と amfid

これは、XNUのコード署名検証の背後にあるロジックを提供し、システム上で実行されるコードの整合性を強制することに焦点を当てています。また、権限を確認し、デバッグを許可したり、タスクポートを取得したりするなどの他の敏感なタスクを処理することもできます。

さらに、一部の操作では、kextはユーザースペースで実行されているデーモン `/usr/libexec/amfid` に連絡することを好みます。この信頼関係は、いくつかの脱獄で悪用されてきました。

AMFIは**MACF**ポリシーを使用し、起動時にフックを登録します。また、その読み込みやアンロードを防ぐと、カーネルパニックが発生する可能性があります。ただし、AMFIを弱体化させるいくつかのブート引数があります：

* `amfi_unrestricted_task_for_pid`: 必要な権限なしに task\_for\_pid を許可
* `amfi_allow_any_signature`: 任意のコード署名を許可
* `cs_enforcement_disable`: コード署名の強制を無効にするためのシステム全体の引数
* `amfi_prevent_old_entitled_platform_binaries`: 権限のあるプラットフォームバイナリを無効にする
* `amfi_get_out_of_my_way`: amfiを完全に無効にする

これらは、登録されるMACFポリシーの一部です：

* **`cred_check_label_update_execve:`** ラベルの更新が行われ、1が返されます
* **`cred_label_associate`**: AMFIのmacラベルスロットをラベルで更新
* **`cred_label_destroy`**: AMFIのmacラベルスロットを削除
* **`cred_label_init`**: AMFIのmacラベルスロットに0を移動
* **`cred_label_update_execve`:** プロセスの権限を確認し、ラベルの変更が許可されるべきかを判断します。
* **`file_check_mmap`:** mmapがメモリを取得し、実行可能として設定しているかを確認します。その場合、ライブラリの検証が必要かどうかを確認し、必要であればライブラリ検証関数を呼び出します。
* **`file_check_library_validation`**: ライブラリ検証関数を呼び出し、プラットフォームバイナリが別のプラットフォームバイナリを読み込んでいるか、プロセスと新しく読み込まれたファイルが同じTeamIDを持っているかなどを確認します。特定の権限により、任意のライブラリを読み込むことも許可されます。
* **`policy_initbsd`**: 信頼されたNVRAMキーを設定
* **`policy_syscall`**: バイナリに制限のないセグメントがあるか、環境変数を許可するかなど、DYLDポリシーを確認します...これは、`amfi_check_dyld_policy_self()`を介してプロセスが開始されるときにも呼び出されます。
* **`proc_check_inherit_ipc_ports`**: プロセスが新しいバイナリを実行する際、他のプロセスがプロセスのタスクポートに対してSEND権を持っている場合、それを保持するかどうかを確認します。プラットフォームバイナリは許可され、`get-task-allow`権限がそれを許可し、`task_for_pid-allow`権限が許可され、同じTeamIDを持つバイナリも許可されます。
* **`proc_check_expose_task`**: 権限を強制
* **`amfi_exc_action_check_exception_send`**: 例外メッセージがデバッガに送信されます
* **`amfi_exc_action_label_associate & amfi_exc_action_label_copy/populate & amfi_exc_action_label_destroy & amfi_exc_action_label_init & amfi_exc_action_label_update`**: 例外処理中のラベルライフサイクル（デバッグ）
* **`proc_check_get_task`**: `get-task-allow`のような権限を確認し、他のプロセスがタスクポートを取得できるか、`task_for_pid-allow`が許可されているかを確認します。どちらもない場合、`amfid permitunrestricteddebugging`を呼び出して許可されているかを確認します。
* **`proc_check_mprotect`**: `mprotect`がフラグ`VM_PROT_TRUSTED`で呼び出された場合、拒否します。これは、その領域が有効なコード署名を持っているかのように扱われるべきことを示します。
* **`vnode_check_exec`**: 実行可能ファイルがメモリに読み込まれるときに呼び出され、`cs_hard | cs_kill`を設定します。これにより、ページが無効になるとプロセスが終了します。
* **`vnode_check_getextattr`**: MacOS: `com.apple.root.installed`と`isVnodeQuarantined()`を確認
* **`vnode_check_setextattr`**: get + com.apple.private.allow-blessおよび内部インストーラー相当の権限
* &#x20;**`vnode_check_signature`**: 権限、信頼キャッシュ、および`amfid`を使用してコード署名を確認するためにXNUを呼び出すコード
* &#x20;**`proc_check_run_cs_invalid`**: `ptrace()`呼び出し（`PT_ATTACH`および`PT_TRACE_ME`）をインターセプトします。`get-task-allow`、`run-invalid-allow`、および`run-unsigned-code`のいずれかの権限があるかを確認し、いずれもない場合はデバッグが許可されているかを確認します。
* **`proc_check_map_anon`**: mmapが**`MAP_JIT`**フラグで呼び出された場合、AMFIは`dynamic-codesigning`権限を確認します。

`AMFI.kext`は他のカーネル拡張のためのAPIも公開しており、その依存関係を見つけることが可能です：
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

これは、`AMFI.kext`がユーザーモードでコード署名をチェックするために使用するユーザーモードのデーモンです。\
`AMFI.kext`がデーモンと通信するためには、特別なポート`18`である`HOST_AMFID_PORT`を介してmachメッセージを使用します。

macOSでは、特別なポートをrootプロセスがハイジャックすることはもはや不可能であり、これらは`SIP`によって保護されており、launchdのみがそれらを取得できます。iOSでは、応答を返すプロセスが`amfid`のCDHashをハードコーディングしていることがチェックされます。

`amfid`がバイナリをチェックするように要求されたときとその応答を見ることが可能であり、デバッグして`mach_msg`にブレークポイントを設定することで確認できます。

特別なポートを介してメッセージが受信されると、**MIG**が呼び出されている関数に各関数を送信するために使用されます。主要な関数は逆アセンブルされ、本書内で説明されています。

## Provisioning Profiles

プロビジョニングプロファイルは、コードに署名するために使用できます。コードに署名してテストするために使用できる**Developer**プロファイルと、すべてのデバイスで使用できる**Enterprise**プロファイルがあります。

アプリがApple Storeに提出され、承認されると、Appleによって署名され、プロビジョニングプロファイルはもはや必要ありません。

プロファイルは通常、拡張子`.mobileprovision`または`.provisionprofile`を使用し、次のコマンドでダンプできます：
```bash
openssl asn1parse -inform der -in /path/to/profile

# Or

security cms -D -i /path/to/profile
```
Although sometimes referred as certificated, these provisioning profiles have more than a certificate:

* **AppIDName:** アプリケーション識別子
* **AppleInternalProfile**: これをApple内部プロファイルとして指定
* **ApplicationIdentifierPrefix**: AppIDNameの前に追加される（TeamIdentifierと同じ）
* **CreationDate**: `YYYY-MM-DDTHH:mm:ssZ`形式の日付
* **DeveloperCertificates**: Base64データとしてエンコードされた（通常は1つの）証明書の配列
* **Entitlements**: このプロファイルに許可されている権利
* **ExpirationDate**: `YYYY-MM-DDTHH:mm:ssZ`形式の有効期限
* **Name**: アプリケーション名、AppIDNameと同じ
* **ProvisionedDevices**: このプロファイルが有効なUDIDの配列（開発者証明書用）
* **ProvisionsAllDevices**: ブール値（企業証明書の場合はtrue）
* **TeamIdentifier**: アプリ間の相互作用の目的で開発者を識別するために使用される（通常は1つの）英数字の文字列の配列
* **TeamName**: 開発者を識別するために使用される人間が読める名前
* **TimeToLive**: 証明書の有効期間（日数）
* **UUID**: このプロファイルのユニバーサルユニーク識別子
* **Version**: 現在1に設定されています

権利のエントリには制限された権利のセットが含まれ、このプロビジョニングプロファイルはAppleのプライベート権利を与えないように特定の権利のみを与えることができます。

プロファイルは通常`/var/MobileDeviceProvisioningProfiles`にあり、**`security cms -D -i /path/to/profile`**で確認することができます。

## **libmis.dyld**

これは`amfid`が何かを許可すべきかどうかを尋ねるために呼び出す外部ライブラリです。これは、すべてを許可するバックドア付きのバージョンを実行することによって、脱獄で歴史的に悪用されてきました。

macOSでは、これは`MobileDevice.framework`内にあります。

## AMFI Trust Caches

iOS AMFIは、アドホックに署名された既知のハッシュのリストを維持しており、これを**Trust Cache**と呼び、kextの`__TEXT.__const`セクションにあります。非常に特定の敏感な操作では、外部ファイルでこのTrust Cacheを拡張することが可能です。

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
