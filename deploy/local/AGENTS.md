# Notes for agents

- This directory is the fork's local deployment (it was the strata-local repository). Run `make` here, in
  `deploy/local`; the image is built from the repository root's working tree, so an uncommitted change to the engine,
  the server or the console is what `make up` deploys.
- Shut down the deployment (`make down`) before performing any build: an image build here, or a build of this
  repository's engine for testing. Bring it back with `make up` when done.
- The deployment runs on pc26's rootless Podman over SSH (`CONTAINER_CONNECTION` in `mise.local.toml`), so `make`,
  `podman` and `podman compose` act on pc26, not this machine; image builds upload the repository and compile there.
