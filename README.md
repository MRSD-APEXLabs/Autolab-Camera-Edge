# Autolab-Camera-Edge: camera hub

HTTP server for the Jetson AGX Xavier (192.168.1.101). It streams **one** of the two stereo cameras at a
time as unrectified 960x600 JPEG pairs, using the wire format of the old Nano server
(`scripts/zedx_nano_v4l2_stream.py`). The Thor rectifies the pairs and computes depth the same way for either
camera.

| name        | camera                          | backend                                                  | role  |
|-------------|---------------------------------|----------------------------------------------------------|-------|
| `zedx`      | ZED X, SN 42757821              | ZED SDK (pyzed), SVGA 60 fps, sent at 30 fps, no depth   | base  |
| `zedx_nano` | ZED X Nano, SN 99292912         | raw V4L2 Bayer (`/dev/video3,/dev/video2`), 30 fps       | wrist |

The Nano is read raw because it needs ZED SDK >= 5.3, which JetPack 5 does not have. Its software AE, white
balance and demosaicing are the old server's code, plus two fixes:
- The exposure and gain carry over across restarts and switches, so there are no dark frames after a switch.
- A 1 MB pipe from v4l2-ctl gives 28-30 pairs/s instead of about 22.

The ZED X runs in a child process, `zedx_capture.py`, which the hub starts on each activation and reads over a
pipe. pyzed holds the Python GIL while it opens (~5 s) and closes (~0.6 s) the camera. Inside the hub, that
froze every request, even `/status`, on each switch. The child process also holds the SDK's memory, and the hub
can kill it if the SDK hangs.

This repo is plain Python 3.8 (stdlib `http.server`, numpy, OpenCV), not a ROS package (see `COLCON_IGNORE`).
On the Xavier it lives in `/home/autolab/camera_hub/`.

The old servo/inspect code (YOLO, IBVS, xArm control, WebSocket servers) is kept in `deprecated/` for reference.
The hub does not use it.

## API (port 8090)

| request | answer |
|---------|--------|
| `GET /` | Page with the active camera, switch buttons, a live left/right preview and stats |
| `GET /status` | `active` (`zedx`, `zedx_nano` or null), `state` (`idle`, `starting`, `streaming` or `error`), `error`, `switch_count`, per-camera `fps`/`frames`/`restarts`/`last_error`. Per-camera `activation` and the hub's `started_ns` change whenever open feeds were ended, so a client can tell it must reconnect |
| `POST /select` `{"camera": "zedx"}` | Stops the other camera, then starts this one. Waits for the first frame. Returns 200, 502 (start failed; the hub keeps retrying), 504 (timeout), 409 (another selection came in while waiting), 503 (the hub is shutting down) or 400 (unknown camera). `null`/`"none"` releases both. `?wait=0` returns at once and `?timeout=30` sets the wait |
| `GET /time` | `{"t_ns", "monotonic_ns"}` for clock sync |
| `GET /cameras/<name>/info`, `/calibration.conf`, `/stats` | Always available (`info` uses the old `/info` schema plus `camera`, `active`, `hub_active`) |
| `GET /cameras/<name>/video_feed/<left\|right\|stereo>` | MJPEG. Stereo parts carry the old headers plus `X-Camera` and `X-Serial`. The ZED X is hardware-synced, so it sends `X-Sync-Us: 0` |
| `GET /cameras/<name>/snapshot/<left\|right>.jpg` | Latest frame |
| Legacy `/info`, `/calibration.conf`, `/video_feed[/<eye>]`, `/snapshot.jpg`, `/snapshot/<eye>.jpg`, `/stats` | The old Nano server, unchanged, for `teleop/nano_stream.py` and `act_inference --camera-source nano-stream`. These routes serve the Nano only |

A camera that is not active answers its feeds and snapshots with **503** JSON:
`{"error": "camera zedx is not active", "camera": "zedx", "active": "zedx_nano"}`. The same happens on legacy
routes while the ZED X is active. Deselecting a camera closes its open streams within about 1 s, so clients
reconnect instead of freezing.

```bash
curl http://192.168.1.101:8090/status
curl -X POST -H 'Content-Type: application/json' -d '{"camera": "zedx"}' http://192.168.1.101:8090/select
curl -o left.jpg http://192.168.1.101:8090/cameras/zedx/snapshot/left.jpg
```

## Run

Deploy from the Thor with `./deploy.sh`. It rsyncs this repo to the Xavier, minus `.git`, `.venv`, `deprecated/`,
`docs/` and other dev files. It keeps these runtime files there:
- `state.json`
- logs
- cached `SN*.conf`

Then start the hub on the Xavier:

```bash
cd ~/camera_hub && python3 camera_hub.py --port 8090     # --help lists the camera options
# in the background:
cd ~/camera_hub; setsid nohup python3 camera_hub.py --port 8090 > hub.log 2>&1 < /dev/null & echo $! > hub.pid
kill $(cat ~/camera_hub/hub.pid)                          # stops cleanly (SIGTERM releases the camera)
```

The selection is saved in `state.json`, so the hub reopens the last camera after a restart. On the very first
start it opens the Nano (`--default-camera`).

## Service

`camera-hub.service` runs the hub at boot. Install it once, as a sudoer on the Xavier:
1. Stop a hub you started by hand.
2. Keep the old `zedx-nano-stream.service` disabled, because it uses the same port and the same Nano devices.
3. Run `~/camera_hub/install_service.sh`.

The unit starts after `driver_zed_loader`, `zed_x_daemon` and `nvargus-daemon`, but deliberately does not
`Want=` them, because pulling in the loader restarts nvargus and kills SDK sessions. To pick up new code after a
deploy, run `sudo systemctl restart camera-hub.service`. Logs are in `journalctl -u camera-hub`.

## Switching

Switching takes these times on the Xavier:
- To the Nano: about 1.8-1.9 s.
- To the ZED X: about 6.2 s. Most of it is `sl.Camera.open()` (~5.5 s); starting `zedx_capture.py` and
  importing pyzed adds ~0.6 s.

The hub answers all requests during a switch. While a camera is starting, `/status` shows `starting`. A switch
away from the ZED X while it is still opening waits for the open to finish, because the SDK cannot abort it. Open
streams of the old camera end right away. A stream of the new camera never gets a frame from its previous
activation. The Nano drops its first 5 frames after each
start, because they were exposed with the sensor's dark defaults before the carried-over exposure applied.

## Troubleshooting

- **`state: error`**: `error` says why, and the hub retries every 2-5 s until the camera works or you select
  another one. Common causes:
  - ZED X `open did not finish within 30 s` or `sent no frames for 5.0 s`. The SDK hung; the hub killed
    `zedx_capture.py` and reopens the camera.
  - ZED X `CAMERA_NOT_DETECTED` or open failures. Another process holds the camera (check `ps aux | grep -i zed`;
    the hub's own `zedx_capture.py` runs only while the ZED X is selected, and exits when the hub does),
    or `zed_x_daemon` is not running.
  - ZED X `open failed: FAILURE` within about 20 s of a hub restart. `nvargus-daemon` was still tearing down the
    old process's session (`journalctl -u nvargus-daemon` shows `CameraProvider destroyed` for the old pid). The
    retry 2 s later opens the camera.
  - Nano `Device or resource busy`. Another v4l2-ctl or the old service holds `/dev/video2`/`/dev/video3`.
- **Port 8090 in use**: the old service or a second hub is running (`ss -ltnp 'sport = :8090'`).
- **Thor client gets 503**: the other camera is active. Select the right one on the page or with `POST /select`.
- **Nano too dark or bright**: check `mean`, `exposure_us` and `gain` in `/cameras/zedx_nano/stats`. AE converges
  in about 1 s. `--nano-ae-target` sets the target brightness.
- **No calibration (404)**: the ZED X reads `/usr/local/zed/settings/SN42757821.conf`. The Nano reads
  `SN99292912.conf` here, which `deploy.sh` copies from `~/zedx_nano_stream/`. Otherwise the hub downloads the
  file from calib.stereolabs.com and caches it.
- Do not restart `driver_zed_loader`, `zed_x_daemon` or `nvargus-daemon` to "fix" the ZED X while it streams. A
  restart kills the SDK session, and the hub reopens the camera by itself once they are back.

## Tests

The tests use fake sources, a fake `v4l2-ctl` and a fake `pyzed` (`tests/fake_pyzed/`, also loaded by
`zedx_capture.py` processes through `PYTHONPATH`), with no hardware:
- From a checkout of this repo: `python -m pytest -q -p no:cacheprovider tests`
- On the Xavier: `cd ~/camera_hub && python3 -m pytest -q -p no:cacheprovider tests`

`test_nano_source.py::test_session_streams_both_eyes_and_stop_kills_the_children` reads
`/proc/sys/fs/pipe-max-size`, so it passes only on Linux.
