# Process supervision

Real gap this closes: the live server has been a manually-started
`nohup ... &` process all session -- no auto-restart on a crash, no
auto-start after this WSL machine reboots. A crash currently means the
site is down until someone notices and manually restarts it.

`observe-api.service` is a real systemd unit (this WSL distro has
systemd running as PID 1, confirmed via `systemctl --version` and
`ps -p 1`). It runs the exact same command that's been used manually all
session (`python3 -m uvicorn server:app --host 0.0.0.0 --port 8000`,
same working directory, same `.env` loaded via `EnvironmentFile=` --
systemd's own mechanism for this, not a `source`d shell subprocess),
with `Restart=always` and `RestartSec=5`.

I can't install this myself -- no passwordless sudo in this WSL
instance, and installing a system service is a real, standing change to
how the machine behaves on every future boot, worth you running
directly rather than me running blind under sudo.

## Install (one time)

```bash
# Stop the manually-started process first -- systemd's own copy will
# otherwise fail to bind :8000 (already in use)
kill $(pgrep -f "uvicorn server:app")

sudo cp /home/gbran/observe-api/deploy/observe-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now observe-api
```

## After install

```bash
systemctl status observe-api        # confirm it's running
sudo journalctl -u observe-api -f   # live logs (server.log still gets everything too)
```

A code deploy from here on is just:
```bash
sudo systemctl restart observe-api
```
instead of the manual kill/nohup dance used earlier this session.

## Real, disclosed limitation

This only supervises the process *inside* WSL. If the WSL instance
itself stops (Windows restarts, WSL is shut down, the host machine
sleeps/reboots without WSL auto-start configured), the service doesn't
come back until WSL itself is running again -- that's a Windows/WSL
boot-behavior question, not something this unit file can reach into.
Not fixed here; worth knowing about rather than assuming this makes the
deployment fully reboot-proof.
