# EchoSync — Oracle Cloud Always Free ARM64 (Ampere A1) Deployment Runbook

Target: **Ubuntu 24.04 (or 22.04) aarch64** on an **Ampere A1 Flex** shape
(Always Free: up to 4 OCPUs / 24 GB RAM), Docker Compose, Caddy reverse proxy
with automatic TLS.

```
                      ┌───────────────────────────────────────────────────┐
 Client (browser)     │  OCI VM (Ampere A1, ubuntu-24.04-aarch64)         │
 wss://voice.example  │                                                   │
        │             │  ┌─────────┐   bridge net   ┌─────────────────┐   │
        │  :80/:443   │  │  caddy  │ ──────────────►│  echosync app   │   │
        └────────────►│  │  (TLS)  │   app:8000     │  (no host ports)│   │
   Security List /    │  └─────────┘                └────────┬────────┘   │
   NSG: 80,443 only   │       ▲  caddy_data volume           │ models vol │
                      └───────┼──────────────────────────────┼────────────┘
                              │ TLS certs                    │ VAD/STT/TTS weights
                        internet (Gemini API, model downloads)
```

The app container publishes **no host ports**. Only Caddy binds 80/443, and it
proxies to `app:8000` over an isolated Docker bridge network.

---

## 1. Provision the VM (OCI Console)

1. **Compute → Instances → Create Instance**
   - Shape: **VM.Standard.A1.Flex**, 4 OCPU / 24 GB (Always Free max) or
     1 OCPU / 6 GB minimum.
   - Image: **Canonical Ubuntu 24.04 aarch64** (or 22.04). Do not pick an
     x86 image.
   - Boot volume: 47 GB (models ≈ 400 MB + Caddy certs + logs).
2. Note the **public IPv4** and, if you want a hostname, create a DNS **A
   record** (e.g. `voice.example.com`) pointing at it.
3. SSH key auth only; disable password login (OCI default).

---

## 2. VCN / Security List (L4 firewall, applied *before* the host)

In the OCI Console: **Networking → Virtual Cloud Networks → your VCN →
Security Lists → Default Security List → Add Ingress Rules**:

| Stateless | Source | Protocol | Dest port | Purpose |
|---|---|---|---|---|
| No | `0.0.0.0/0` | TCP | 80 | HTTP + ACME HTTP-01 challenge + redirect |
| No | `0.0.0.0/0` | TCP | 443 | HTTPS (+ WebSocket upgrade) |
| No | `0.0.0.0/0` | UDP | 443 | HTTP/3 (QUIC) — optional |
| No | `<your-ip>/32` | TCP | 22 | SSH — restrict to your IP, never `0.0.0.0/0` |

Egress: keep the default **allow-all egress** — the app needs outbound HTTPS
(Gemini API, GitHub/HuggingFace model downloads, Let's Encrypt) and EdgeTTS.

Equivalent **OCI CLI** (replace placeholders):

```bash
oci network security-list update \
  --security-list-id <ocid1.securitylist...> \
  --ingress-security-rules \
    '[{"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":80,"max":80}}},
      {"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":443,"max":443}}},
      {"source":"<your-ip>/32","protocol":"6","tcpOptions":{"destinationPortRange":{"min":22,"max":22}}}]' \
  --force
```

If you use **Network Security Groups** instead, attach the same rules to an NSG
on the instance's VNIC. Both layers are enforced at the hypervisor — nothing you
do on the host can bypass them, and closing a port here hides it from the
internet entirely.

> **Always Free tip:** the shape is flexible — start at 4 OCPU / 24 GB, then
> downscale via **Instance → Edit → Shape** if the app idles.

---

## 3. Host preparation (Docker on Ubuntu aarch64)

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Official Docker CE repo (arm64 packages are first-class)
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"   # re-login afterwards

# Verify native arm64 toolchain
docker version --format '{{.Server.Os}}/{{.Server.Arch}}'   # => linux/arm64
docker buildx ls                                          # => linux/arm64*
```

Optional but recommended on the 6 GB shape — 2 GB swap so first-boot model
downloads never OOM:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 4. Deploy the stack

```bash
git clone <repo-url> echosync && cd echosync

cp .env.production.example .env.production
$EDITOR .env.production
#   REQUIRED: GEMINI_API_KEY
#   Set DOMAIN to your DNS name (or the VM's public IP / localhost)
#   Set WS_ALLOWED_ORIGINS to exactly ["https://<DOMAIN>"]

# Build natively on ARM64 (no emulation) and start everything.
# --env-file supplies ${DOMAIN} for the Caddyfile; the app also gets the file
# injected as container env vars.
docker compose --env-file .env.production up -d --build
```

First boot downloads the neural weights (~350 MB: Silero VAD, faster-whisper,
Kokoro) into the `models` Docker volume and pre-warms the models — expect
2–5 minutes before `/ready` turns green (the healthcheck `start_period` is 300 s
for exactly this reason). Later restarts are instant and skip the downloads.

---

## 5. Firewall on the host (defense in depth)

OCI's Security List is the real perimeter; a host firewall adds a second layer.
**Only Caddy publishes ports (80/443) through Docker**, so allowing those two
ports on the host covers everything.

### Option A — ufw (recommended)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp  comment 'HTTP + ACME'
sudo ufw allow 443/tcp comment 'HTTPS + WSS'
sudo ufw allow 443/udp comment 'HTTP/3 QUIC (optional)'
sudo ufw --force enable
sudo ufw status verbose
```

> **Docker + ufw caveat:** Docker publishes container ports via its own iptables
> chains (`DOCKER`), which are evaluated *before* ufw's `ufw-user-input` chain.
> This is safe here **because the `app` service publishes no host ports** — the
> only Docker-published ports are Caddy's 80/443, which you are intentionally
> exposing anyway. Never add `-p 8000:8000` to the app service while ufw is your
> only guard.

### Option B — raw iptables + persistence

```bash
# Rate-limited HTTP/HTTPS, SSH allowed
sudo iptables -A INPUT -i lo -j ACCEPT
sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80  -m limit --limit 25/min --limit-burst 100 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/min --limit-burst 100 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 443 -j ACCEPT   # HTTP/3 (optional)
sudo iptables -P INPUT DROP

# Persist across reboots
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
sudo iptables -L INPUT -n -v --line-numbers   # verify
```

Do **not** add ACCEPT rules for 8000/tcp — the app must only be reachable
through Caddy on the internal bridge network.

---

## 6. Verify the deployment

```bash
# 1. Both containers Up and (app) healthy — wait for "healthy", not just "Up"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
# Expected:
#   NAMES             STATUS                     PORTS
#   echosync-caddy    Up ... (healthy)           0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp, 443/udp
#   echosync-app      Up ... (healthy)           8000/tcp            <- no host mapping = correct

# 2. Health status detail
docker inspect --format '{{json .State.Health}}' echosync-app | python3 -m json.tool

# 3. Liveness through the proxy (host + public)
curl -fsS http://127.0.0.1/healthz                      # {"status":"ok"}
curl -fsS https://<DOMAIN>/healthz

# 4. Readiness — all three models must be true, HTTP 200
curl -fsS https://<DOMAIN>/ready
# {"status":"ready","models":{"vad":true,"stt":true,"tts":true},...}

# 5. Prometheus scrape endpoint
curl -fsS https://<DOMAIN>/metrics | head

# 6. WebSocket handshake (wss through Caddy)
#    From another machine:
sudo apt-get install -y wscat     # or: npx wscat
wscat -c "wss://<DOMAIN>/ws/audio" --header "Origin: https://<DOMAIN>"
# Expect: {"type":"status",...,"state":"LISTENING"} — 101 Switching Protocols

# 7. TLS certificate actually issued by Let's Encrypt
echo | openssl s_client -connect <DOMAIN>:443 -servername <DOMAIN> 2>/dev/null | \
  openssl x509 -noout -issuer -dates
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `echosync-app` stuck `starting` > 5 min | Model download slow/blocked | `docker compose logs -f app`; check outbound HTTPS; weights resume idempotently |
| Caddy logs `502 Bad Gateway` | App still warming or crashed | `docker compose logs app`; `/healthz` locally via `docker exec echosync-app python -c ...` |
| Caddy never gets LE cert | Ports 80/443 blocked at VCN, or no A record | Check Security List; `caddy logs`; LE staging rate limits |
| WS rejected immediately (close code 4403) | Origin not allowlisted | `WS_ALLOWED_ORIGINS` must exactly match `https://<DOMAIN>` |
| `401` on every WS connect | `WS_BEARER_TOKEN` set to empty string | Keep it commented out in `.env.production` |
| OOM kill during first boot | < 6 GB RAM, no swap | Add swap (§3); use `WHISPER_MODEL_NAME=tiny.en` |

### Routine operations

```bash
docker compose --env-file .env.production logs -f app     # JSON logs (rotated 10m x3)
docker compose --env-file .env.production pull && \
  docker compose --env-file .env.production up -d --build # update
docker run --rm -v echosync_models:/m alpine du -sh /m    # model cache size
tar czf backup-$(date +%F).tar.gz caddy_data models       # back up certs + weights
```
