# host-switch

Adds `192.168.0.55  computer.jimmyhmiller.com` to `/etc/hosts` when you're on your
home network, and removes it everywhere else. Runs at boot, every 30 seconds, and
immediately whenever the network changes. Zero dependencies — pure Rust `std`.

## How "home" is detected

Not by Wi-Fi SSID — on macOS 14+/26 the SSID reads back as `<redacted>` unless the
process holds Location Services permission, which a background daemon can't get.

Instead it fingerprints home by the **default gateway's MAC address** (your router's
hardware address: `bc:07:1d:75:e2:1c`). Stable, survives reboots, needs no permission.
It's read via the base-OS `route` and `arp` commands.

## Install

```sh
sudo ./install.sh
```

This builds the release binary, copies it to `/usr/local/bin/host-switch`, installs the
LaunchDaemon at `/Library/LaunchDaemons/com.jimmyhmiller.host-switch.plist`, and starts it.
Root is required because it edits `/etc/hosts`.

## Check / debug

```sh
host-switch status         # show detected gateway MAC + decision (no changes made)
tail -f /var/log/host-switch.log
```

## Uninstall

```sh
sudo ./uninstall.sh        # stops daemon, strips the managed block, removes files
```

## Changing the config

Edit the constants at the top of `src/main.rs`:

- `HOME_GATEWAY_MAC` — your router's MAC. Find it while on the network with:
  `arp -n "$(route -n get default | awk '/gateway/{print $2}')"`
- `MANAGED_LINES` — the `/etc/hosts` line(s) to install when home.

Then re-run `sudo ./install.sh`.

The managed block is delimited by markers so the rest of your `/etc/hosts` is never touched:

```
# >>> host-switch (managed) >>>
192.168.0.55	computer.jimmyhmiller.com
# <<< host-switch (managed) <<<
```
