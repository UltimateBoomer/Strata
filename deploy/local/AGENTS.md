# Notes for agents

- Shut down the deployment (`make down`) before performing any build: an image build here, or a build of the Strata
  fork (`~/Projects/Strata`) for testing. Bring it back with `make up` when done.
- The deployment runs on pc26's rootless Podman over SSH (`CONTAINER_CONNECTION` in `mise.local.toml`), so `make`,
  `podman` and `podman compose` act on pc26, not this machine; image builds upload this checkout and compile there.
