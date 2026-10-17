# Deploying Theraflo Ads to LocalStack for Azure: a practice guide

You will deploy this project the way you would deploy it to Microsoft Azure, using the same
`az` commands, but against **LocalStack for Azure**, an emulator on your own PC. Nothing is
billed and nothing touches a real Azure account.

Every step has the same parts:

- **Run**: the command;
- **Why**: what it is for;
- **Parameters**: what each flag means;
- **Check**: how to see that it worked;
- **On real Azure**: what would be different.

There are two stages. Do Stage 1 first, on its own.

| | What runs | Time |
|---|---|---|
| **Stage 1** | The dev console as one container app (in-memory backend) | about 45 minutes |
| **Stage 2** | The full architecture: managed Postgres, Redis, RabbitMQ, the API, the outbox relay, 4 workers, and jobs | 2 to 3 hours, with some exploring |

All commands are **PowerShell**, run from `C:\Users\Maboiidevs\Desktop\theraflo\theraflo-ads`.

> **Container Apps won't start? Use Part B-ACI, or try the volume fix.** On Windows, the
> emulator's Container Apps runtime never starts its Kubernetes cluster. The log shows
> `Error starting K3D cluster … Timeout while waiting for EKS startup` three times, then
> `is FAILED, not ACTIVE`.
>
> **The cause** (seen with `LOCALSTACK_DEBUG=1`): the emulator gives k3d a Windows folder
> (`C:\Users\…\tmp\azure-files:/mnt/azure-files`) to mount. k3d splits mount specs at `:`, so the
> `C:` breaks the spec, and k3d exits in under a second with `Failed Cluster Configuration
> Validation … volume mount destination doesn't appear to be an absolute path`. The emulator
> doesn't notice and waits 180 s before reporting a timeout. It isn't a slow download.
>
> - **Part B-ACI** runs Stage 1 on **Azure Container Instances**, a simpler service backed by plain
>   Docker containers. It needs no cluster.
> - **The volume fix (not yet tested end to end):** name the same folder in a form without a drive
>   letter, which Docker Desktop also accepts. Add this line to the `[[containers]]` block of
>   `.lstk/config.toml`, then `lstk stop` and `lstk start` (this erases the emulator's state):
>   ```toml
>   volume = "/run/desktop/mnt/host/c/Users/Maboiidevs/AppData/Local/lstk/volume/localstack-azure"
>   ```
>   `lstk` also creates an empty, unused `C:\run\desktop\…` folder, which you can delete.
>
> **Windows, before Part A:** if `lstk setup azure` fails with `'C:\Program' is not recognized`,
> make a space-free alias for the Azure CLI, then put it first on `PATH` in every terminal that
> runs `lstk az`:
> ```powershell
> New-Item -ItemType Junction -Path "$env:USERPROFILE\azcli" -Target "C:\Program Files\Microsoft SDKs\Azure\CLI2"
> ```
> ```powershell
> $env:Path = "$env:USERPROFILE\azcli\wbin;" + $env:Path
> ```

---

## 0. The ideas you will use

| Azure concept | What it is | In our `docker-compose.yml` |
|---|---|---|
| **Subscription** | The billing and permission boundary that everything lives in. LocalStack gives you a fake one, all zeros. | none |
| **Resource group** | A folder of resources. Deleting the group deletes everything in it. | none |
| **Container registry (ACR)** | A private Docker registry that stores your images | your local `docker images` |
| **Container Apps environment** | The shared space apps run in: a private network, internal DNS, and shared logging | the Docker network compose creates |
| **Container app** | A long-running container with optional ingress (an address) and replicas (copies) | one service, such as `api` or `relay` |
| **Revision** | An immutable version of a container app. Each change to its image or settings creates one. | a container recreated by `up` |
| **Replica** | One running copy of a revision | one container (`--scale` copies) |
| **Ingress** | How traffic reaches an app: `external` (from outside), `internal` (other apps only), or none | `ports:`, or none |
| **Job** | A container that runs to completion and stops, manually or on a schedule | `migrate`, `deletions` |
| **Secret** | A value stored apart from the app's settings; settings refer to it by name | values in `.env` |
| **Azure Database for PostgreSQL (Flexible Server)** | Postgres run for you: backups, patching, high availability | the `postgres` service |

One image runs every role. API, relay, workers, and jobs all run the same `theraflo-ads` image
with a different command (`serve`, `relay`, `worker accounting`, `migrate`, and so on), exactly as in
compose.

---

## 1. Factors and limitations: read before starting

### The emulator is not Azure

These features are **stored but not enforced** (from LocalStack's documentation):

| Feature | In the emulator | Consequence for us |
|---|---|---|
| Autoscaling and scale-to-zero | Apps run a fixed number of replicas (`--min-replicas`, at least 1) | `--min-replicas 0` won't scale to zero; set 1 |
| Traffic splitting across revisions | Stored; the latest revision gets all traffic | You can practise the commands but not see a split |
| Scheduled and event jobs | Stored but **never fire** | Start every job by hand with `job start` |
| Azure Files volumes | Skipped: the container starts without the mount | `models/`, `reports/`, and `data/` can't be mounted; bake models into the image (step B5b) |
| Dapr | No sidecar | Not used by us |
| Certificates and custom domains | Not issued or verified | Use the generated addresses |
| `az containerapp logs show --follow` and `exec` | Not supported | Use `kubectl` (section C3) |
| Postgres firewall, private networking, HA, backups | Stored, not enforced | Real Azure needs firewall rules and TLS |

### Practical factors

| Factor | Detail |
|---|---|
| **Cost** | $0. LocalStack for Azure is free during its public preview, until general availability (expected early 2027). Pricing after that is not announced. |
| **Account** | A free LocalStack account with the **Azure preview plan** active. The emulator does not start without a valid auth token. |
| **Nothing persists** | The Azure emulator has no persistence or snapshots. `lstk stop`, a reboot, or a Docker restart erases every resource. Keep your commands in a script so you can replay them. |
| **Memory** | Docker has about 7.8 GB. The first container app in an environment starts a local Kubernetes cluster (k3d) inside Docker. Stage 1 fits beside your compose stack; for Stage 2, stop the compose stack. |
| **Disk** | The image is 1.27 GB and is stored three times: locally, in the emulated registry, and in the k3d cluster. Keep at least 5 GB free. |
| **Ports** | The emulator listens on `127.0.0.1:4566` (and 443 if free). |
| **DNS** | Registry and app addresses end in `.localhost.localstack.cloud`, which public DNS resolves to `127.0.0.1`. Some routers and DNS filters block such answers. See Troubleshooting. |
| **Security** | The console hands out dev credentials, and the emulator's portal has no real sign-in. Keep everything on `127.0.0.1`, which is the default. Never publish port 4566 to your network. |
| **Real Azure safety** | Always type `lstk az …`, never plain `az …`. `lstk az` uses its own Azure CLI configuration pointed at the emulator. Plain `az` would act on a real subscription if you ever signed in to one. |

---

## Part A. One-time setup

### A1. LocalStack account

**Do:** sign up at app.localstack.cloud, activate the **Azure preview plan** on the pricing page,
and find your auth token in the dashboard.
**Why:** the emulator checks a licence when it starts. The token is personal: do not paste it into
chats, scripts, or files you share.

### A2. Install the Azure CLI

**Run:**
```powershell
winget install -e --id Microsoft.AzureCLI
```
**Why:** the `az` CLI is what you would use against real Azure. LocalStack does not replace it; it
redirects it.
**Check:** open a new terminal and run `az version`.

### A3. Install `lstk`

**Run:**
```powershell
npm install -g @localstack/lstk
```
**Why:** `lstk` signs you in, downloads and starts the emulator container, and runs `az` against
it (`lstk az …`).
**Check:** `lstk --version`.

### A4. Sign in

**Run:**
```powershell
lstk login
```
**Why:** it stores your token in the Windows credential store, so later starts are licensed.
**Alternative:** set `$env:LOCALSTACK_AUTH_TOKEN` yourself in the terminal, and never in a file you share.

### A5. Configure the emulator, with the portal turned on

**Do:** create `theraflo-ads/.lstk/config.toml`:
```toml
[[containers]]
type = "azure"
tag  = "latest"
port = "4566"
env  = ["portal"]

[env.portal]
LS_AZURE_PORTAL = "1"
```
**Why:**
- **`type = "azure"`:** without it, `lstk start` starts the AWS emulator.
- **`port = "4566"`:** the edge port. With no host part, it is published on `127.0.0.1` only. Keep it that way.
- **`LS_AZURE_PORTAL = "1"`:** turns on the local Azure Portal emulator, a preview that is off by default.

`lstk` writes logs and an Azure CLI configuration next to this file. If you ever put this folder in
git, commit only `config.toml` (`.lstk/*` ignored, `!.lstk/config.toml` kept).

### A6. Start the emulator

**Run** (from `theraflo-ads`, so `lstk` finds the config):
```powershell
lstk start
```
If it doesn't pick up the file, run `lstk --config .lstk/config.toml start`.

**Why:** it pulls `localstack/localstack-azure` (about 210 MB the first time) and starts it.
**Check:** `docker ps` shows the emulator container. Open the portal at
http://localhost:4566/_localstack/portal/ (the sign-in is a one-click mock).

### A7. Connect the Azure CLI to the emulator

**Run:**
```powershell
lstk setup azure
```
**Why:** it registers a "LocalStack" cloud in a separate Azure CLI configuration and signs in with
dummy credentials. From now on, `lstk az <command>` goes to the emulator, and your normal `az`
configuration is untouched.

There is an alternative, `lstk az start-interception`, which redirects **plain** `az` globally
until you run `stop-interception`. It is convenient, but forgetting to stop it is how people end
up confused about which Azure they are talking to. This guide uses the `lstk az` prefix.

### A8. Install the Container Apps CLI extension

**Run:**
```powershell
lstk az extension add --name containerapp --upgrade
```
**Why:** `az containerapp …` commands live in an extension, not in the core CLI.

### A9. Check where you are pointing

**Run:**
```powershell
lstk az account show -o table
```
**Check:** the subscription id is `00000000-0000-0000-0000-000000000000`. If you ever see a real
subscription here, stop.

---

## Part B. Stage 1: the console as a container app

What you are building:
```
your browser ──https──▶ LocalStack gateway :4566 ──▶ Container Apps environment (k3d)
                                                     └─ theraflo-console (1 replica)
                                                         image pulled from ──▶ ACR therafloacr
```

### B1. Names you will reuse

**Run:**
```powershell
$RG = "theraflo-rg"; $LOC = "westeurope"; $ACA_ENV = "theraflo-env"; $ACR = "therafloacr"
```
**Why:** later commands refer to these. Registry names must be globally unique on real Azure,
5 to 50 characters, lowercase letters and digits only. The location doesn't matter in the emulator;
on real Azure it decides latency, price, and data residency.

### B2. Resource group

**Run:**
```powershell
lstk az group create --name $RG --location $LOC
```
**Why:** every resource needs a group. You can remove the whole deployment in one command (section E).
**Check:** `lstk az group show --name $RG -o table`, or look under Resource groups in the portal.
**On real Azure:** groups are also where you assign permissions and tags for cost reporting.

### B3. Container registry

**Run:**
```powershell
lstk az acr create --name $ACR --resource-group $RG --location $LOC --sku Basic --admin-enabled true
```
**Why:** Container Apps pull images from a registry. Your local Docker images aren't visible to the
cloud, or to the emulator's cluster.

**Parameters:**

| Flag | Meaning |
|---|---|
| `--sku Basic` | Tier: Basic, Standard, or Premium. The tier decides storage, throughput, geo-replication, and private links. Basic is enough for learning. |
| `--admin-enabled true` | Turns on a single username and password for the registry. Simple. |

**On real Azure:** leave the admin user off. Give the container app a **managed identity** with the
`AcrPull` role (`az containerapp create … --registry-identity system`), so no password exists to leak.

### B4. Registry address and login

**Run** each line, then check with `$LOGIN`:
```powershell
$LOGIN = lstk az acr show --name $ACR --resource-group $RG --query loginServer -o tsv
```
```powershell
$ACR_USER = lstk az acr credential show --name $ACR --query username -o tsv
```
```powershell
$ACR_PASS = lstk az acr credential show --name $ACR --query "passwords[0].value" -o tsv
```
```powershell
$ACR_PASS | docker login $LOGIN --username $ACR_USER --password-stdin
```
**Why:**
- **`loginServer`** is the registry's hostname. Here it looks like
  `therafloacr.azurecr.azure.localhost.localstack.cloud:4566`; on real Azure, `therafloacr.azurecr.io`.
- **`--query … -o tsv`** extracts one field as plain text: JMESPath query, tab-separated output. You
  will use this pattern constantly.
- **`--password-stdin`** keeps the password out of your command history.

**Check:** `Login Succeeded`. An informational warning before it is expected: the emulator doesn't
reproduce Azure's full sign-in flow.

### B5. Build the image

**B5a. The standard image:**
```powershell
docker compose --profile app build api
```
**Why:** compose builds `theraflo-ads-api:latest` from the `Dockerfile`, which is multi-stage and
runs as a non-root user (uid 10001).

**B5b. Optional, recommended: add the model files.** `.dockerignore` excludes `models/`, and the
emulator skips Azure Files mounts, so without this the Ranker lab's ML ranker says "no model".
Create `deploy/localstack-azure/Dockerfile`:
```dockerfile
FROM theraflo-ads-api:latest
COPY --chown=app:app . /home/app/models
```
and build it **with `models` as the build context**. The last argument is the context: the only
files `COPY` can see.
```powershell
docker build -t theraflo-ads-api:with-models -f deploy/localstack-azure/Dockerfile models
```
**Why this way:** a small image layered on top of the real one. The project's `.dockerignore`
doesn't apply, because the context is the `models` folder, and the main `Dockerfile` stays
unchanged.
**On real Azure:** this is a reasonable way to ship a model: the version is pinned to the image,
and rollback means deploying the previous image tag. The alternative is mounting a file share or
downloading the model at startup.

### B6. Push the image

**Run:**
```powershell
$IMAGE = "$LOGIN/theraflo-ads:v1"
```
```powershell
docker tag theraflo-ads-api:with-models $IMAGE
```
```powershell
docker push $IMAGE
```
(If you skipped B5b, tag `theraflo-ads-api:latest` instead.)

**Why:**
- A tag's first part names the registry, so `docker push` knows where to send the image.
- **Tag versions (`v1`, `v2`…), never `latest`.** Every deploy then names exactly what runs, and a
  rollback means deploying the old tag.

**Check:** `lstk az acr repository list --name $ACR -o table` shows `theraflo-ads`.

### B7. Container Apps environment

**Run:**
```powershell
lstk az containerapp env create --name $ACA_ENV --resource-group $RG --location $LOC --logs-destination none
```
**Why:** apps must live in an environment. Apps in the same environment share a private network
and can call each other by name.

**Parameters:**

| Flag | Meaning |
|---|---|
| `--logs-destination none` | Don't create a Log Analytics workspace. On real Azure, Log Analytics is billed per GB, and chatty JSON access logs (ours) add up. Alternatives: `log-analytics`, `azure-monitor`. |

**On real Azure:**
- **Workload profiles.** Choose between the Consumption plan (pay per second) and Dedicated
  workload profiles (reserved machines, plus a management fee).
- **Your own VNet.** Bring one (`--infrastructure-subnet-resource-id`) to reach private databases
  and to expose TCP ports, which we need in Stage 2.

### B8. Create the container app

**Run:**
```powershell
lstk az containerapp create --name theraflo-console --resource-group $RG --environment $ACA_ENV --image $IMAGE --registry-server $LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS --ingress external --target-port 8000 --allow-insecure --cpu 1 --memory 2Gi --min-replicas 1 --max-replicas 1 --revision-suffix v1 --env-vars THERAFLO_ADS_REVIEWER=simulated THERAFLO_ADS_AUTH_FALLBACK_HEADER=on
```
**Why:** this is the deployment. The first app in an environment also creates the emulator's k3d
cluster, so this can take several minutes. Pulling the 1.27 GB image into the cluster takes a few
more.

**Parameters:**

| Flag | Meaning, and why this value |
|---|---|
| `--image` | What to run: the tag you pushed |
| `--registry-*` | How the platform authenticates its image pull (admin credentials here; a managed identity on real Azure) |
| `--ingress external` | Give the app an address reachable from outside the environment. `internal` would make it reachable only by other apps. |
| `--target-port 8000` | The port **inside** the container that ingress forwards to. Our image's command is `theraflo-ads serve --host 0.0.0.0 --port 8000`. Getting this wrong is the most common reason a new app returns 502. |
| `--allow-insecure` | Accept plain HTTP as well as HTTPS. Real Azure otherwise redirects HTTP to HTTPS, which you'd keep for real deployments. |
| `--cpu 1 --memory 2Gi` | Resources per replica. Container Apps only accepts set pairs with **memory = 2 × CPU**: 0.25/0.5Gi, 0.5/1Gi, … 2/4Gi. 2 GiB leaves room for the console's jobs, which start a second Python process. |
| `--min-replicas 1 --max-replicas 1` | **Exactly one copy.** The in-memory backend keeps state in the process: with two replicas, each would have its own users, decisions, and review queue, and the console would flip between them. Scale-out is only safe with shared state (Stage 2). The emulator also doesn't support scale-to-zero. |
| `--revision-suffix v1` | Names the revision `theraflo-console--v1`, readable in listings |
| `--env-vars THERAFLO_ADS_REVIEWER=simulated` | App setting: turns on the free simulated reviewer, so submitted creatives get reviewed automatically |
| `THERAFLO_ADS_AUTH_FALLBACK_HEADER=on` | **Emulator only.** The emulator's ingress deletes the `Authorization` header (see Part F), so the API also reads the token from `X-Theraflo-Authorization`, which the console sends too. Real Azure forwards `Authorization`, so leave this off there; the app refuses it in production. |

Not set, because the image's defaults are what we want:
- **`THERAFLO_ADS_BACKEND`:** memory.
- **`THERAFLO_ADS_ENV`:** development, which mounts the console.
- **The signing keys:** the memory backend generates random ones at startup.

**Check:**
```powershell
lstk az containerapp show --name theraflo-console --resource-group $RG --query "{state:properties.provisioningState, running:properties.runningStatus, revision:properties.latestRevisionName}" -o table
```
`Succeeded` and `Running`.

### B9. Open it

**Run:**
```powershell
$FQDN = lstk az containerapp show --name theraflo-console --resource-group $RG --query properties.configuration.ingress.fqdn -o tsv
```
```powershell
start "https://${FQDN}:4566/ui/"
```
**Why:** the FQDN is `<app>.<environment's default domain>`. LocalStack serves every app through its
gateway on port 4566, with a certificate it trusts. If HTTPS fails, use `http://` (you set
`--allow-insecure`).

**Check:** the console loads with **memory backend** in the top bar. Try the App simulator and the
Ranker lab. Jobs that need `data/` show "missing", which is expected: that folder isn't in the image.

**On real Azure:** the address would be `https://theraflo-console.<random>.<region>.azurecontainerapps.io`
and **public on the internet**. Never deploy the dev console that way. Restrict ingress to your IP
(`az containerapp ingress access-restriction set …`), or put Microsoft sign-in in front of it
(`az containerapp auth …`).

---

## Part B-ACI. Stage 1 on Azure Container Instances

Use this when Container Apps can't start its cluster (see the note at the top). Steps **B1 to B6**
are the same: resource group, registry, login, image push. Do those first; you then have `$RG`,
`$LOGIN`, `$ACR_USER`, `$ACR_PASS`, and `$IMAGE`. Skip B7 to B9.

### What Container Instances is, compared with Container Apps

| | Container Instances (ACI) | Container Apps (ACA) |
|---|---|---|
| The unit | A **container group**: one or more containers sharing one IP address, `localhost`, and lifecycle (like a Kubernetes pod) | An **app** with revisions and replicas inside an environment |
| Addresses | A public or private IP, plus an optional DNS name. **HTTP only**: no built-in HTTPS | Ingress with an HTTPS address, internal or external |
| Scaling | None: one group is one copy | Replicas, autoscaling, scale-to-zero |
| Updates | Most changes mean **delete and recreate** | New revisions, with traffic splitting |
| Jobs | Use `--restart-policy Never` or `OnFailure` | Real Jobs (manual, scheduled, event) |
| Billing on real Azure | Per second while running: about $0.05/hour for 1 vCPU + 2 GB. No free monthly allowance. | Per second, with a free monthly allowance and scale-to-zero |
| Good for | Simple single services, batch tasks, CI | Microservices, APIs, background workers |

**How LocalStack runs it** (checked on this machine):
- Each container group becomes a **plain Docker container** named `ls-aci-<group>-<container>`.
- Its ports are **published on random ports of your PC**.
- The Azure-style DNS name (`<label>.<region>.azurecontainer.io`) is stored but **not reachable**
  from your PC. You use `http://localhost:<random port>` instead.

### ACI1. Create the container group

**Run:**
```powershell
lstk az container create --name theraflo-console --resource-group $RG --image $IMAGE --registry-login-server $LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS --os-type Linux --cpu 1 --memory 2 --ports 8000 --ip-address Public --dns-name-label theraflo-console-practice --restart-policy OnFailure --environment-variables THERAFLO_ADS_REVIEWER=simulated
```

**Why:** this is the whole deployment: one container group with one container, pulled from your
registry. It takes about a minute, mostly copying the 1.27 GB image out of the emulated registry.

**Parameters:**

| Flag | Meaning, and why this value |
|---|---|
| `--image $IMAGE` | The tag you pushed in B6 |
| `--registry-login-server` / `--registry-username` / `--registry-password` | Registry credentials for the image pull. Note the flag name differs from Container Apps (`--registry-server`). On real Azure, prefer a managed identity: `--assign-identity` plus `--acr-identity`. |
| `--os-type Linux` | Our image is Linux. ACI also runs Windows containers on real Azure; LocalStack runs Linux only. |
| `--cpu 1 --memory 2` | Cores and **GB as plain numbers** (Container Apps wants `2Gi`). Unlike Container Apps, any reasonable mix is allowed. 2 GB leaves room for the console's jobs. |
| `--ports 8000` | Which port of the container to expose. ACI has **no port mapping**: the exposed port must be the one the app listens on. Our image runs `serve --port 8000`. |
| `--ip-address Public` | Give the group a reachable IP. `Private` needs a VNet, which LocalStack doesn't emulate. |
| `--dns-name-label …` | On real Azure: `theraflo-console-practice.westeurope.azurecontainer.io`, which must be unique in the region. In LocalStack it's stored but not reachable; practise it anyway. |
| `--restart-policy OnFailure` | `Always` (the default) restarts even after a clean exit. `OnFailure` restarts only after a crash. `Never` is for run-once tasks, ACI's version of a job. A server should run forever, so either of the first two works; `OnFailure` shows that the choice exists. |
| `--environment-variables` | Plain settings, **visible** in `az container show`. Use `--secure-environment-variables KEY=value` for secrets: their values are hidden in output. |

**Check:**
```powershell
lstk az container show --name theraflo-console --resource-group $RG --query instanceView.state -o tsv
```
`Running`. Then look at the real container behind it:
```powershell
docker ps --filter "label=ls-aci-group=theraflo-console" --format "{{.Names}}  {{.Ports}}"
```
Expect `ls-aci-theraflo-console-theraflo-console  0.0.0.0:<HOSTPORT>->8000/tcp`.

### ACI2. Open it

Find the random host port, then open the console:
```powershell
$CID = docker ps -q --filter "label=ls-aci-group=theraflo-console"
```
```powershell
$HOSTPORT = (docker port $CID 8000 | Select-Object -First 1).Split(":")[-1]
```
```powershell
start "http://localhost:$HOSTPORT/ui/"
```

**Why plain HTTP:** ACI only gives a container an IP and ports, with no HTTPS. On real Azure
you'd add HTTPS with a reverse-proxy sidecar container in the same group (Caddy or nginx with a
certificate), or put Application Gateway in front.

**Check:** the console loads with **memory backend** in the top bar.

**Security, important:** LocalStack publishes the port on **all** network interfaces (`0.0.0.0`),
not only `127.0.0.1`. Other devices on your network might reach the console, which hands out dev
credentials, if Windows Firewall lets them. Stop or delete the group when you're not using it
(ACI3, ACI5), and don't do this on public Wi-Fi.

### ACI3. Operate it (practise these)

**Logs** work here, unlike in Container Apps:
```powershell
lstk az container logs --name theraflo-console --resource-group $RG --container-name theraflo-console
```

**Stop, start, restart.** Stopping releases the compute; on real Azure, billing stops too.
```powershell
lstk az container stop --name theraflo-console --resource-group $RG
```
```powershell
lstk az container start --name theraflo-console --resource-group $RG
```
```powershell
lstk az container restart --name theraflo-console --resource-group $RG
```
After a start or restart:
- **Data resets:** the in-memory data starts empty.
- **New port:** the host port may change, so repeat ACI2.

**Events and state history:**
```powershell
lstk az container show --name theraflo-console --resource-group $RG --query "containers[0].instanceView" -o json
```

### ACI4. Ship a new version

`az container update` doesn't exist in LocalStack, and on real Azure most properties can't be
changed in place either. Deploying a new version means **delete, then create again**, with the new
image tag or settings:

1. Build and push `…/theraflo-ads:v2` (as in B5 and B6).
2. Delete the group:
   ```powershell
   lstk az container delete --name theraflo-console --resource-group $RG --yes
   ```
3. Run ACI1 again with `--image "$LOGIN/theraflo-ads:v2"`.

**Why this matters:** each new version means downtime between the delete and the next start.
That's the main reason real services use Container Apps (revisions, rolling updates) or
Kubernetes, and use ACI for simple or batch work.

### ACI5. Clean up

```powershell
lstk az container delete --name theraflo-console --resource-group $RG --yes
```
Or remove everything, including the registry: section E (`lstk az group delete`).

### Limitations of LocalStack's Container Instances

| Limitation | Effect |
|---|---|
| No `az container exec` or `attach` | Use `docker exec -it ls-aci-theraflo-console-theraflo-console sh` instead |
| Linux containers only | Fine for us |
| No VNet integration | `--ip-address Private` and subnets can't be practised |
| No `az container update` | Delete and recreate (also true for most properties on real Azure) |
| DNS name not reachable | Use `localhost:<host port>` |
| Port published on all interfaces | See the security note in ACI2 |
| Random host ports can be reserved by Windows | Hyper-V and WSL reserve port ranges (`netsh interface ipv4 show excludedportrange protocol=tcp`). If LocalStack draws one, the create fails with `ports are not available … forbidden by its access permissions` and leaves a container behind. Clean up and retry (section F). |
| No persistence | The emulator forgets everything on restart, as with every service |

### Next step: the full stack on Container Instances (optional)

A container group's containers share `localhost`, so one group could hold the API, relay,
workers, Redis, and RabbitMQ together (like a single Kubernetes pod), with Postgres as the
emulated flexible server (D1). It's defined in a YAML file (`az container create --file
group.yaml`). This teaches multi-container groups, sidecars, and secure environment variables. It
isn't how you'd run production, because nothing scales separately, but it's a good exercise.

---

## Part C. Operating what you deployed (practise these)

Part C is for **Container Apps** (Part B). If you deployed with Part B-ACI, there's no k3d cluster;
use ACI3 instead.

### C1. Status, revisions, and replicas
```powershell
lstk az containerapp revision list --name theraflo-console --resource-group $RG -o table
```
```powershell
lstk az containerapp replica list --name theraflo-console --resource-group $RG -o table
```

### C2. Ship a new version (a new revision)
Change something visible. For example, switch the reviewer to rules-only:
```powershell
lstk az containerapp update --name theraflo-console --resource-group $RG --set-env-vars THERAFLO_ADS_REVIEWER=none --revision-suffix v2
```
**Why:** any change to the image or settings creates a new **revision**. Run C1 again to see `v2`.
- **Single mode (the default):** the new revision takes all traffic and the old one is deactivated.
- **`--revisions-mode multiple`:** revisions run side by side with traffic weights (`az containerapp
  ingress traffic set`). The emulator stores the weights but sends everything to the latest revision.
- **Watch out:** a new revision means a new process, so the in-memory data starts empty again.

### C3. Logs, through Kubernetes
`az containerapp logs show` isn't emulated. Container Apps here run on a k3d cluster, which ships
with `kubectl`.
Find the cluster's server container. If this prints nothing, no cluster is running (see the note at
the top about k3d on Windows):
```powershell
$K3D = docker ps --format "{{.Names}}" | Where-Object { $_ -like "k3d-*-server-0" } | Select-Object -First 1; $K3D
```
List every pod, with its namespace:
```powershell
docker exec $K3D kubectl get pods -A
```
Copy the `NAMESPACE` and `NAME` of the app's pod from that table into two variables, then read its
logs:
```powershell
$NS = "paste-namespace-here"; $POD = "paste-pod-name-here"
```
```powershell
docker exec $K3D kubectl logs -n $NS $POD --tail 50
```
**Why variables, not `<placeholders>`:** PowerShell reads `<` as an operator, so a pasted
`<pod-name>` fails with `The '<' operator is reserved for future use`.
**Why it's worth seeing:** on real Azure this is hidden, but Container Apps *is* Kubernetes
underneath. Apps are deployments, replicas are pods, and ingress is an Envoy proxy.

### C4. Secrets
```powershell
lstk az containerapp secret set --name theraflo-console --resource-group $RG --secrets demo-secret=hello
```
```powershell
lstk az containerapp update --name theraflo-console --resource-group $RG --set-env-vars DEMO_VALUE=secretref:demo-secret --revision-suffix v3
```
**Why:** `secretref:` makes a setting read its value from a secret. Secret values are not shown by
`az containerapp show` or in the portal; plain environment variables are. On real Azure, secrets
can also reference **Key Vault** (`--secrets name=keyvaultref:<uri>,identityref:<identity>`), so
rotation happens in one place.

### C5. Look at everything in the portal
http://localhost:4566/_localstack/portal/ (click **Sign in**; no password, you're "Local
Developer"): your resource group, registry, environment, and apps or container groups.
- **Activity log** (left menu) lists every control-plane operation your `az` commands made, such as
  `Microsoft.ContainerInstance/containerGroups/write`: the REST calls behind the CLI.
- **JSON View** on a resource's Overview shows the raw resource that `az … show` returns.
- **Greyed-out items** (Metrics, Logs, Alerts, CLI / PS) are cloud-only; hover for the reason.

---

## Part D. Stage 2: the full architecture

What you are building:
```
browser ─▶ theraflo-api (external) ─┬─▶ Postgres flexible server  (decisions, ledger, outbox, profiles…)
                                    ├─▶ theraflo-redis (internal TCP 6379)   counters, budgets, bandit counts
                                    └─▶ outbox rows ─▶ theraflo-relay ─▶ theraflo-rabbitmq (internal TCP 5672)
                                                                         ├─▶ theraflo-accounting (billing)
                                                                         ├─▶ theraflo-features (features, bandits)
                                                                         └─▶ theraflo-analytics (data lake)
                                    theraflo-review ─▶ Postgres (creative review queue)
jobs: theraflo-migrate (schema), theraflo-seed (demo inventory)
```

**Order matters:**
1. Infrastructure first: Postgres, Redis, RabbitMQ.
2. Then the **migrate** job, which creates the schema.
3. Then **seed**, which loads the demo campaigns.
4. Then the API.
5. Then the relay and workers.

Apps that start before their dependencies crash-loop until those exist. That's harmless, but noisy.

**Memory budget.** The cluster node can reserve about 7.3 GiB, and Kubernetes places pods by what
they *reserve* (`--memory`), not by what they use. Measured, the API and console each use well under
100 MiB but reserve 2 GiB in Stage 1. A new revision also needs room **beside** the old one until it's
ready. Stage 2 fits only with smaller reservations, so before D5 shrink the Stage 1 console:
```powershell
lstk az containerapp update --name theraflo-console --resource-group $RG --cpu 0.5 --memory 1Gi
```

This stage relies on emulator behaviour the docs don't describe, mainly app-to-app TCP and
resolving the Postgres address from inside the cluster. Each such step has a **checkpoint**.

### D0. Free up memory and port 5432
```powershell
docker compose --profile app --profile monitoring stop
```
**Why:** your compose stack uses about 1 GB of memory and port 5432. Stopping keeps its data;
`docker compose … start` brings it back later.

### D1. Postgres flexible server

Choose a test password of **letters and digits only**: symbols such as `@`, `:` or `/` would have to
be URL-encoded inside the database URL. Type it at a prompt so it stays out of your history:
```powershell
$PG_PASS = Read-Host "Postgres test password"
```
```powershell
lstk az postgres flexible-server create --name theraflo-pg --resource-group $RG --location $LOC --admin-user theraflo --admin-password $PG_PASS --sku-name Standard_B1ms --tier Burstable --version 16 --storage-size 32 --public-access All --yes
```

**Parameters:**

| Flag | Meaning |
|---|---|
| `--tier Burstable --sku-name Standard_B1ms` | The smallest, cheapest compute (1 vCore, 2 GiB, burstable CPU). The tiers are Burstable, then GeneralPurpose, then MemoryOptimized. |
| `--version 16` | Major Postgres version. Ours runs 16 in compose too: keep dev and prod on the same major version. |
| `--storage-size 32` | GiB. On real Azure, storage can grow but never shrink. |
| `--public-access All` | Allow connections from any IP. **Emulator only:** firewall rules aren't enforced here anyway. On real Azure use private access (VNet integration) or specific IPs, never `All`. |
| `--yes` | Don't stop for interactive confirmation prompts |

```powershell
lstk az postgres flexible-server db create --server-name theraflo-pg --resource-group $RG --name theraflo_ads
```
```powershell
$PG_HOST = lstk az postgres flexible-server show --name theraflo-pg --resource-group $RG --query fullyQualifiedDomainName -o tsv
```
**Check:** in the emulator, `$PG_HOST` looks like `theraflo-pg.postgres.database.localhost.localstack.cloud:4512`: the emulator's gateway forwards that port to the Postgres container. On real Azure it's `theraflo-pg.postgres.database.azure.com`.

**On real Azure:**
- **Encryption is required** (`require_secure_transport` is on). Our database driver (pg8000) would
  need an SSL context; that's a small code change, not needed in the emulator.
- **Use separate accounts.** Use a non-admin database user for the app, and keep the admin
  password in Key Vault.

### D2. Redis and RabbitMQ as container apps

**Why containers and not managed services:**
- LocalStack doesn't emulate Azure Cache for Redis.
- Azure has no managed RabbitMQ. Service Bus would mean rewriting our queue code.
- Our Redis data can be rebuilt, so a plain container is acceptable. RabbitMQ in a single container
  without a volume loses queued messages on restart. That's tolerable here, because the outbox relay
  re-sends anything not yet published, but on real Azure you'd give it persistent storage.

```powershell
lstk az containerapp create --name theraflo-redis --resource-group $RG --environment $ACA_ENV --image redis:7-alpine --ingress internal --transport tcp --target-port 6379 --exposed-port 6379 --cpu 0.25 --memory 0.5Gi --min-replicas 1 --max-replicas 1
```
```powershell
lstk az containerapp create --name theraflo-rabbitmq --resource-group $RG --environment $ACA_ENV --image rabbitmq:3.13-alpine --env-vars RABBITMQ_DEFAULT_USER=theraflo RABBITMQ_DEFAULT_PASS=theraflo --ingress internal --transport tcp --target-port 5672 --exposed-port 5672 --cpu 0.5 --memory 1Gi --min-replicas 1 --max-replicas 1
```

**Parameters:**

| Flag | Meaning |
|---|---|
| `--ingress internal` | Reachable only from apps in the same environment, never from outside |
| `--transport tcp` | Raw TCP instead of HTTP. Redis and AMQP aren't HTTP. |
| `--exposed-port` | The port other apps connect to, at `<app-name>:<exposed-port>`. On real Azure, TCP ingress needs an environment in **your own VNet**. The emulator ignores it: apps connect through a Kubernetes service on port 80 instead (D3). |
| `--max-replicas 1` | One copy. Several independent Redis or RabbitMQ containers would split the data. |

**Checkpoint 1:**
- **Rejected flag:** if `--transport tcp` or `--exposed-port` is rejected, the emulator doesn't
  support TCP ingress. Fall back to running Redis and RabbitMQ as plain Docker containers, and point
  the apps at `host.k3d.internal:<port>`.
- **Image pull fails** (Docker Hub rate limits): push the two images to your registry, the same way
  as in B6, and use those tags instead.

### D3. Keys, secrets, and shared settings

Generate three random 32-byte keys, one per line:
```powershell
$TOKEN_KEY = .venv/Scripts/python -c "import secrets; print(secrets.token_hex(32))"
```
```powershell
$HASH_KEY = .venv/Scripts/python -c "import secrets; print(secrets.token_hex(32))"
```
```powershell
$SESSION_KEY = .venv/Scripts/python -c "import secrets; print(secrets.token_hex(32))"
```
What the three keys do:
- **Token key:** signs impression tokens, so clicks can't be forged.
- **Hash key:** hashes user ids, so no raw ids are stored.
- **Session key:** signs test-user sessions.

Every API, relay, and worker replica must use the **same** keys, which is why they're generated
once and shared (as in `.env`).

```powershell
$SECRETS = @("token-key=$TOKEN_KEY", "hash-key=$HASH_KEY", "session-key=$SESSION_KEY")
```

**Emulator only: Redis's and RabbitMQ's addresses.** Real Azure gives every app a name inside the
environment, so the settings would say `theraflo-redis:6379` and `theraflo-rabbitmq:5672`. The
emulator doesn't: those names never resolve (and the cluster's DNS server restarts under the failed
lookups). Each app is a Kubernetes service named after its **revision**, in its own namespace, on
port 80. Look the two up (after D2):
```powershell
$K3D = docker ps --format "{{.Names}}" | Where-Object { $_ -like "k3d-*-server-0" } | Select-Object -First 1
```
```powershell
$svc = docker exec $K3D kubectl get svc -A --no-headers -o custom-columns=NAME:.metadata.name,NS:.metadata.namespace
```
```powershell
$REDIS_HOST = $svc | Select-String "theraflo-redis" | ForEach-Object { $n, $ns = -split $_.Line; "$n.$ns.svc.cluster.local" }
```
```powershell
$AMQP_HOST = $svc | Select-String "theraflo-rabbitmq" | ForEach-Object { $n, $ns = -split $_.Line; "$n.$ns.svc.cluster.local" }
```
Each prints one name like `rev-1c005a2db6.aca-app-theraflo-redis-….svc.cluster.local`. If Redis or
RabbitMQ ever gets a new revision, its name changes: look it up again, rebuild `$APP_ENV`, and update
the apps.
```powershell
$APP_ENV = @("THERAFLO_ADS_BACKEND=infra", "THERAFLO_ADS_DATABASE_URL=postgresql+pg8000://theraflo:${PG_PASS}@${PG_HOST}/theraflo_ads", "THERAFLO_ADS_REDIS_URL=redis://${REDIS_HOST}:80/0", "THERAFLO_ADS_AMQP_URL=amqp://theraflo:theraflo@${AMQP_HOST}:80/%2F", "THERAFLO_ADS_LAKE_PATH=/data/lake", "THERAFLO_ADS_TOKEN_KEY=secretref:token-key", "THERAFLO_ADS_USER_HASH_KEY=secretref:hash-key", "THERAFLO_ADS_SESSION_KEY=secretref:session-key", "THERAFLO_ADS_REVIEWER=simulated", "THERAFLO_ADS_AUTH_FALLBACK_HEADER=on")
```
**Check before going on.** A missing key or setting doesn't fail here; it fails later, inside a
container. Expect `10`, then `64 64 64`:
```powershell
$APP_ENV.Count
```
```powershell
$SECRETS | ForEach-Object { $_.Split("=", 2)[1].Length }
```

**Why this shape:**
- **`@APP_ENV`, not `$APP_ENV`, in the commands below.** `--env-vars` and `--secrets` expect each
  item as its own argument. `lstk` is a PowerShell script (`lstk.ps1`) that forwards its arguments,
  and a `$`-array passed through it arrives as **one** argument with the items joined by spaces.
  Azure then stores a single setting, `THERAFLO_ADS_BACKEND`, whose value is all the others; the
  app doesn't recognise `infra …`, so it quietly runs the in-memory build. Writing `@APP_ENV`
  (*splatting*) passes the items separately. Check after creating an app:
  `lstk az containerapp show … --query "properties.template.containers[0].env[].name"` lists ten
  names.
- **Keys go in as secrets,** referenced with `secretref:`. The database password sits inside the
  URL, which is a plain setting. That's acceptable for practice; on real Azure put the whole URL in
  a secret, backed by Key Vault.
- **`%2F`** is the URL-encoded `/`: RabbitMQ's default "virtual host".
- **No port after `${PG_HOST}`.** The emulator's server address already ends in one
  (`…localhost.localstack.cloud:4512`); real Azure's (`….postgres.database.azure.com`) has none, and
  the driver then uses Postgres' default, 5432. Adding `:5432` breaks the emulator's URL
  (`invalid literal for int() … '4512:5432'`).
- **The lake is a folder in the container,** so its files disappear on restart (no volumes in the
  emulator). On real Azure, mount an Azure Files share there.

### D4. Jobs: create the schema, then load demo data

**Why jobs:** migrations must run **once, before** the apps, and **never concurrently**. Running
them inside the API's startup would race when there are several replicas. A job runs to completion
and stops, and it can be re-run safely: our migrations are idempotent.

```powershell
lstk az containerapp job create --name theraflo-migrate --resource-group $RG --environment $ACA_ENV --trigger-type Manual --replica-timeout 900 --replica-retry-limit 1 --image $IMAGE --registry-server $LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS --command theraflo-ads --args migrate --secrets @SECRETS --env-vars @APP_ENV --cpu 0.5 --memory 1Gi
```
```powershell
lstk az containerapp job start --name theraflo-migrate --resource-group $RG
```
```powershell
lstk az containerapp job execution list --name theraflo-migrate --resource-group $RG -o table
```

**Parameters:**

| Flag | Meaning |
|---|---|
| `--trigger-type Manual` | Runs only when started. `Schedule` (with `--cron-expression`) and `Event` exist, but the emulator never fires them. |
| `--replica-timeout 900` | Seconds before a run is killed |
| `--replica-retry-limit 1` | Retry once on failure |
| `--command` / `--args` | Override the image's start command: run `theraflo-ads migrate` instead of the server |

Wait for `Succeeded`. Then repeat all three commands with `--name theraflo-seed` and
`--args seed-demo` (seed loads the demo catalog, priors, and profiles; it needs the hash key).

**Checkpoint 2: can the cluster reach Postgres?** It worked on Docker Desktop for Windows: inside a
pod, `$PG_HOST`'s name resolves to the emulator (`172.21.0.4` on the cluster's network), and its
port 4512 answers as Postgres. To see why a run failed, read its pod's log:
`docker exec $K3D kubectl get pods -n aca-jobs`, then
`docker exec $K3D kubectl logs -n aca-jobs <pod-name> --tail 20`.
- **`could not resolve` or `connection refused`** for the server's address: the cluster can't
  reach the emulated server. Run `docker ps` to find the Postgres container the emulator started,
  note its published port, and set `THERAFLO_ADS_DATABASE_URL` to
  `…@host.k3d.internal:<port>/theraflo_ads`. `host.k3d.internal` is the cluster's name for your PC.
  Then rebuild `$APP_ENV` and re-run the job.

### D5. The API

```powershell
lstk az containerapp create --name theraflo-api --resource-group $RG --environment $ACA_ENV --image $IMAGE --registry-server $LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS --ingress external --target-port 8000 --allow-insecure --secrets @SECRETS --env-vars @APP_ENV --cpu 0.5 --memory 1Gi --min-replicas 1 --max-replicas 1 --revision-suffix v1
```
**Why one replica:** two would be safe now, because state lives in Postgres and Redis. But each
process keeps its own Prometheus counters and its own in-process linear bandit. Try
`--max-replicas 2` later as an experiment.

**Check:**
- Open `https://<theraflo-api FQDN>:4566/ui/` (FQDN as in B9). The top bar shows **infra backend**
  and **ready**.
- **Overview → Readiness** lists Postgres, the frequency store, the impression log, and the budget
  store, all green.

**Checkpoint 3:**
- **Red `frequency_store` or `budget_store`:** the API can't reach `theraflo-redis:6379`. Go back to
  Checkpoint 1.
- **Red `postgres`:** go to Checkpoint 2.

### D6. The background processes

They have no ingress: nothing calls them. They pull work from Postgres (relay, review) or RabbitMQ
(workers).

```powershell
lstk az containerapp create --name theraflo-relay --resource-group $RG --environment $ACA_ENV --image $IMAGE --registry-server $LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS --command theraflo-ads --args relay --secrets @SECRETS --env-vars @APP_ENV --cpu 0.25 --memory 0.5Gi --min-replicas 1 --max-replicas 1
```
Repeat with these names and arguments:

| `--name` | `--args` | What it does |
|---|---|---|
| `theraflo-accounting` | `worker accounting` | Bills viewable impressions, clicks, and conversions against budgets |
| `theraflo-features` | `worker features` | Updates user features and the Beta bandit's counts in Redis |
| `theraflo-analytics` | `worker analytics` | Writes decisions and labels to the data lake (training data) |
| `theraflo-review` | `review worker` | Reviews submitted creatives with the configured reviewer |

**Why exactly one relay:** two relays would both publish the same outbox rows. That's safe, because
consumers deduplicate, but wasteful.

**Why workers can scale:** RabbitMQ hands each message to one consumer, so extra worker replicas
share the load.

### D7. Check the whole system

1. **Overview → Start traffic.** Within a minute:
   - **Live** shows decisions per second;
   - **Counters → Client events** show `accepted`.
2. **Ranker lab → Bandit state.** Impressions and clicks per ad grow. This proves the path
   API → outbox → relay → RabbitMQ → features worker → Redis.
3. **Advertiser portal:** submit a Clean Bookly creative. It becomes `approved` within seconds (the
   review worker on Postgres).
4. **App simulator:** click an ad and send a conversion. A few seconds later, the campaign's
   **billed today** rises (the accounting worker).

If step 2 never moves, look at the relay's and features worker's pods (section C3). The usual causes
are the RabbitMQ address, or the relay starting before the migrate job finished; restart it with
`lstk az containerapp revision restart`.

### D8. Optional: your local Qwen reviewer

```powershell
lstk az containerapp update --name theraflo-review --resource-group $RG --set-env-vars THERAFLO_ADS_REVIEWER=ollama THERAFLO_ADS_OLLAMA_URL=http://host.k3d.internal:11434 THERAFLO_ADS_OLLAMA_MODEL=qwen3.5:9b
```
**Why `host.k3d.internal`:** inside the k3d cluster, `localhost` is the pod itself. `host.k3d.internal`
is k3d's name for the machine running Docker, where Ollama listens. If it can't connect, reviews fail
safe into the human queue, by design.

### D9. Optional: the offline experiments in the cloud

**Jobs & reports** runs each experiment as a second process inside the app's container. What works
depends on what's in the image:

| Experiment | In the cloud | Why |
|---|---|---|
| Simulator, reviewer evals (rules, simulated, naive), reviewer comparison, MDE, simulated A/B | Yes | They generate their own data; the models are in the image (B5b) |
| Finalize labels, queued deletions | Only on `theraflo-api` | They need Postgres (the infra backend) |
| Offline RL (FQI), drift monitor | Not with v4 | They read the training log, `data/raw`, which `.dockerignore` keeps out of images |
| Reviewer eval with `ollama` | After D8's settings, on the app that runs it | Ollama runs on your PC, at `host.k3d.internal:11434` |
| Scaling laws (`theraflo-ads research scaling`) | Run it locally | Not a console job: it simulates a 280k-decision log and trains dozens of models; the node has no memory to spare |

To add the training log (30 MB), layer it on the image, then ship it like v3 (preload it, then update):
```powershell
docker build -t theraflo-ads-api:v5 -f deploy/localstack-azure/Dockerfile.data data/raw
```
**Watch the memory:** a job runs inside the app's 1 GiB. If it's killed for running out of memory,
the whole container restarts, and the in-memory console loses its data.

**On real Azure:** research jobs don't run inside the serving app. They run as Container Apps jobs
(like migrate) or on Azure Machine Learning, read their data from Blob Storage, and write reports
back there, so the serving replicas never compete with them for memory.

---

## Part E. Teardown

```powershell
lstk az group delete --name $RG --yes
```
```powershell
lstk stop
```
**Why both:**
- **`group delete`** removes everything you created (practise this: it's how you clean up on real
  Azure).
- **`lstk stop`** stops the emulator. It wipes all state anyway, because the Azure emulator doesn't
  persist.
- **Afterwards:** bring your compose stack back with `docker compose --profile app --profile monitoring start`.
- **Disk space:** `docker image prune` and `docker system df` show what the k3d cluster and the
  emulator left behind.

---

## Part F. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `lstk start` starts the AWS emulator | It didn't find `.lstk/config.toml`. Run it from `theraflo-ads`, or use `--config .lstk/config.toml`, or `--type azure`. |
| Licence or auth error on start | The Azure preview plan isn't active, or the token is missing. Run `lstk login` again. |
| `docker login` or `push`: "no such host" | DNS blocked the `localhost.localstack.cloud` name. Check with `nslookup <the address>`. As administrator, add `127.0.0.1 <the address>` to `C:\Windows\System32\drivers\etc\hosts`. |
| `az containerapp` says the command isn't found | Repeat step A8 (the extension) |
| App creation takes minutes | The first app creates the k3d cluster and pulls 1.27 GB. `docker ps` shows progress. |
| The app's URL returns 502 or 404 | The app isn't ready, or `--target-port` is wrong. Check `runningStatus` and the pod logs (C3). |
| A variable such as `$LOGIN` is empty or has extra text | Print it with `echo $LOGIN`. `lstk az` may print warnings; re-run the command and check you used `-o tsv`. |
| Everything disappeared | The emulator restarted. Re-run from B1 (keep these commands in a `.ps1` script). |
| Container group state `Failed` | Find the cause in the emulator log: `docker logs --since 10m localstack-azure 2>&1 \| Select-String "ACI\|Failed to create"` |
| ACI log: `ports are not available … forbidden by its access permissions` | The random host port is in a range Windows reserves. Remove the leftover container (`docker rm -f ls-aci-<group>-<container>`), delete the group (`lstk az container delete …`), then create it again. |
| ACI log: `Conflict. The container name … is already in use` | A failed earlier attempt left its container behind: same cleanup as the row above |
| Container Apps: `Error starting K3D cluster … Timeout while waiting for EKS startup` | k3d rejects the emulator's Windows-style volume path (`C:\…`) and exits at once; the "timeout" is the emulator waiting 180 s for it. Use Part B-ACI, or the `volume = "/run/desktop/mnt/host/c/…"` fix in the note at the top. To see it yourself, start the emulator with `$env:LOCALSTACK_DEBUG = "1"` and run `docker logs localstack-azure 2>&1 \| Select-String "FATA"`. |
| Every `docker` command: `request returned 500 Internal Server Error for API route … dockerDesktopLinuxEngine` | Docker Desktop's Linux VM stopped, so nothing reaches the engine (`docker version` shows no Server). Restart Docker Desktop (tray icon, Restart); if that hangs, quit it, run `wsl --shutdown`, and start it again. The emulator's state is gone afterwards: `lstk start`, then B1 onwards. |
| First container app fails: `Field 'template.containers….image' is invalid … ImagePullBackOff` | A race on a brand-new cluster: the first pull happens before the emulator has wired up the registry's name inside the cluster, so it goes to `127.0.0.1` and is refused. Kubernetes retries and usually succeeds a minute later, but the emulator has already marked the app `Failed`. Delete the app, wait until its namespace is gone (next row), then create it again: the cluster is ready and the image is cached, so it's fast. **For every new image tag it happens again**, and on `containerapp update` the emulator removes the new revision and keeps the old one. The reliable fix is to preload the image into the cluster node, so no pull is needed (apps use `imagePullPolicy: IfNotPresent`): `docker save --platform linux/amd64 -o $env:TEMP\img.tar $IMAGE`, `docker cp $env:TEMP\img.tar "${K3D}:/tmp/img.tar"`, `docker exec $K3D ctr -n k8s.io images import /tmp/img.tar`. Use a file, not a pipe: Windows PowerShell 5.1 corrupts binary data piped between programs. |
| Create right after a delete: `(403) … namespace aca-app-<app>-… is being terminated` | Kubernetes was still removing the old app (its pods get up to 30 s to stop), and the new app reuses the same namespace name. The emulator may still record the app as `Succeeded`/`Running` with nothing behind it, so delete it again. Wait until `docker exec $K3D kubectl get ns` (C3 sets `$K3D`) no longer lists `aca-app-<app>-…`, then create. |
| Container Apps: the console stays on "Loading the console…" | The emulator reaches apps through the Kubernetes API server's service proxy, which rewrites absolute links in HTML (`/ui/js/main.js` becomes `/api/v1/namespaces/…/proxy/ui/js/main.js`, which the app doesn't serve). The console now uses relative links. An image built before that change has the problem: rebuild (B5), push a new tag (B6), and `containerapp update --image … --revision-suffix …`. |
| Container Apps: the console loads, but most pages fail with `401 invalid credentials` | The emulator's ingress goes through the Kubernetes API server, which authenticates with the `Authorization` header and then deletes it, so your bearer token never reaches the app (a custom header like `X-Advertiser-Key` does). Deploy an image with the console fix and set `THERAFLO_ADS_AUTH_FALLBACK_HEADER=on` (B8). Through the gateway, scripts must send `X-Theraflo-Authorization: Bearer <token>` as well. |
| After `containerapp update`, the browser still shows the old version; the new revision's health is `None` | The new pod is `Pending`: `kubectl get events -n <the app's namespace>` shows `Insufficient memory`. The gateway keeps serving the last **ready** revision. Free reservations (smaller `--cpu`/`--memory`, or delete apps you don't need), then update again with a new `--revision-suffix`. |
| Stage 2 apps run, but the API is the in-memory build, the migrate job fails, and `env[].name` shows only `THERAFLO_ADS_BACKEND` | `$APP_ENV`/`$SECRETS` were passed with `$`: through `lstk.ps1` the list becomes one argument. Use `@APP_ENV`/`@SECRETS` (D3), then fix each app with `containerapp secret set --secrets @SECRETS` and `containerapp update --replace-env-vars @APP_ENV` (jobs: `containerapp job secret set` / `job update`). |
| Migrate job: `ValueError: invalid literal for int() with base 10: '4512:5432'` | `$PG_HOST` already includes the emulator's port. Build the URL as `@${PG_HOST}/theraflo_ads` (D3), then `job update --replace-env-vars @APP_ENV`. |
| API or worker crash-loops: `THERAFLO_ADS_USER_HASH_KEY must be set (hex)` | A key variable was empty when `$SECRETS` was built, so the secret is empty. Run D3's three key lines and its check (`64 64 64`), then `containerapp secret set --secrets @SECRETS` and a new revision. |
| API `/readyz`: `frequency_store` and `budget_store` false, pages slow (about 4 s); CoreDNS restarting; its log shows `theraflo-redis. A: … i/o timeout` | The emulator doesn't resolve app names inside the environment. Use the Kubernetes service names on port 80 (D3, "Emulator only"), rebuild `$APP_ENV`, then `containerapp update --replace-env-vars @APP_ENV`. |
| The PC slows down, or containers are killed (out of memory) | Stop the compose stack, lower `--cpu`/`--memory`, or give Docker Desktop more memory. |

---

## Part G. From practice to real Azure: what changes

| Topic | Emulator | Real Azure |
|---|---|---|
| Cost | $0 | Container Apps per second, Postgres per hour, Log Analytics per GB, registry per day. Set a **budget alert** first. |
| Addresses | `*.localhost.localstack.cloud`, on your PC | Public `*.azurecontainerapps.io`: protect the console, or don't deploy it |
| Image pulls | Registry admin user | Managed identity with `AcrPull` |
| Secrets | Container Apps secrets | Key Vault references, plus managed identity |
| Postgres | No TLS, no firewall | TLS required (a small code change), private access through a VNet, a non-admin app user |
| Redis and RabbitMQ over TCP | Maybe works without a VNet | Needs an environment in your own VNet |
| Scaling | A fixed number of replicas | Real autoscaling, and scale-to-zero for the API (cold starts) |
| Scheduled jobs | Never fire | Deletions hourly and label finalization nightly, as `Schedule` jobs |
| Files | Mounts skipped | Azure Files for the lake and reports, or Blob Storage |
| Logs and metrics | `kubectl logs` | Log Analytics / Azure Monitor, and managed Prometheus and Grafana |
| Repeatability | Commands typed by hand | The same resources described in **Bicep** or Terraform. The natural next exercise: the emulator accepts Bicep through `lstk az deployment group create`. |

---

## Practice exercises

1. **Rollback:** push a `v2` image, deploy it as revision `v2`, then deploy `v1` again by tag.
2. **Break it on purpose:** set a wrong `--target-port`, see the 502, and fix it.
3. **Scale-out with shared state** (Stage 2): set the API to `--min-replicas 2 --max-replicas 2`.
   Does the console still work? Watch two pods appear in `kubectl get pods`.
4. **Fail closed:** delete the `theraflo-redis` app. The API should keep answering, with NO_AD and
   readiness red, never with an error page. Recreate it.
5. **A secret rotation:** change `token-key`, restart the API's revision, and see that old impression
   tokens (clicks on ads served before) are rejected.
6. **Bicep:** describe Stage 1 (group contents: registry, environment, app) in a `main.bicep`, and
   deploy it with `lstk az deployment group create --resource-group $RG --template-file main.bicep`.
