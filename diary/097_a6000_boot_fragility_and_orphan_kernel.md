# Diary 097 — A6000 boot fragility: an orphan kernel in /boot, and where to dig next

Date: 2026-06-09

## What happened

Ralph plugged the A6000 box into a new UPS, which required a cold power
cycle. On first boot afterward, GRUB's default entry panicked:

    Kernel panic - not syncing: VFS: Unable to mount root fs on unknown-block(0,0)

Booting an alternate GRUB entry got the system up. From there we discovered
the box has **two kernels installed and only one of them is real**:

- `6.11.0-29-generic` — Ubuntu HWE kernel; properly installed, works.
- `6.17.0-35-generic` — present in `/boot` (vmlinuz, System.map, config) and
  symlinked from `/boot/vmlinuz` and `/boot/initrd.img`, but **dpkg does
  not track it as a proper `linux-image-*` package**. `sudo update-initramfs
  -u -k all` only regenerated 6.11's initramfs; 6.17 was silently skipped.

GRUB scans `/boot` when `update-grub` runs and adds menu entries for every
`vmlinuz-*` it finds, regardless of dpkg state. So GRUB happily offered
6.17 as the newest kernel and made it the default. The 6.17 initramfs is
either missing, was generated against a wrong/empty module set, or is
otherwise unable to expose the NVMe root device — the kernel comes up,
then panics with the `(0,0)` device-number signature of "no root device
visible at all."

This explains a pattern Ralph reports informally: this box's boot process
has felt fragile for some time. The likely cause isn't anything intrinsic
to the hardware — it's that broken/partial kernels accumulate in `/boot`
and only manifest on a cold boot. While the machine runs, the working
kernel is in RAM and the dormant landmine in `/boot` is invisible.

## The fix applied today

Pinned 6.11.0-29-generic as GRUB's permanent default by **literal menu-entry
name** (not by index, which is fragile across kernel updates). In
`/etc/default/grub`:

    GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-29-generic"

Then `sudo update-grub`. Backup of the original at `/etc/default/grub.bak`.
Verified: clean boot to login, `nvidia-smi` shows the A6000 normally.

This neutralizes the current landmine. A future apt-installed 6.18/6.19/etc.
will NOT shift the default off 6.11 because the anchor is the literal kernel
name. The only way to break boot now is to remove the 6.11.0-29 package
itself (e.g. an `apt autoremove` someday), at which point GRUB_DEFAULT
becomes invalid and falls through to entry 0 — which is, at the moment,
the broken 6.17. So: before ever removing 6.11.0-29, either repair 6.17
first or update GRUB_DEFAULT to whichever working kernel will remain.

## Open question for the A6000 instance to investigate

The fix above is symptomatic. The deeper question is **how 6.17 got installed
in the first place**, because that's the mechanism that will keep planting
landmines if left alone. Ubuntu 24.04 LTS ships 6.8 by default and the HWE
stack moves to 6.11; 6.17 is not in the standard Ubuntu kernel pool. It has
to have come from one of:

- Ubuntu's mainline-kernel PPA (kernel.ubuntu.com/~kernel-ppa/mainline)
- A third-party kernel PPA (xanmod, liquorix, zen)
- A manually-downloaded `.deb` or extracted tarball
- A vendor build (nvidia, OEM)

Diagnostic commands to run on the A6000 box (none are destructive):

    # When was 6.17 first installed and by what mechanism?
    grep -E '6\.17|linux-image' /var/log/apt/history.log* 2>/dev/null
    grep -E '6\.17|linux-image' /var/log/dpkg.log* 2>/dev/null

    # What kernel-related packages does dpkg actually know about?
    dpkg -l | grep -E 'linux-(image|headers|modules)' | awk '{print $1, $2, $3}'

    # What's on disk for each kernel?
    ls -la /boot
    ls /lib/modules/

    # Which apt sources are configured?
    cat /etc/apt/sources.list
    ls /etc/apt/sources.list.d/
    cat /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null

Once the origin is identified there are two clean paths:

1. **Repair 6.17.** If its linux-image and linux-modules `.deb`s are still
   available (or can be reinstalled from the same source), run
   `sudo apt install --reinstall linux-image-6.17.0-35-generic
   linux-modules-6.17.0-35-generic`, then `sudo update-initramfs -c -k
   6.17.0-35-generic`, then test by booting it once via `sudo grub-reboot`.
2. **Remove 6.17 cleanly.** Delete the orphan files in `/boot` (vmlinuz-,
   System.map-, config-, initrd.img-) and any matching `/lib/modules/6.17.0-35-generic/`
   tree, then `sudo update-grub`. If dpkg has any stale state for it,
   `sudo dpkg --purge` the package.

After either, the GRUB_DEFAULT pin should be revisited so it points at
whatever the user actually wants as default going forward.

## A small principle this confirms

`update-grub` will add a menu entry for any vmlinuz it finds in `/boot`,
even one with no working initramfs. This means **the state of `/boot` is a
trust boundary** — anything sitting there is potentially the next boot.
A future habit worth adopting on this box: after any kernel install or
upgrade, verify with `ls -la /boot` that every `vmlinuz-X` has a matching
`initrd.img-X` of plausible size (60–100 MB on modern Ubuntu), and that
`dpkg -l | grep linux-image-X` shows the package as `ii` (installed). If
those two conditions don't both hold, treat the kernel as a landmine and
remove it before the next cold boot.
