# TODO

- Distribution compatibility hardening:
  - Set explicit macOS deployment targets for release builds (for example via `CMAKE_OSX_DEPLOYMENT_TARGET`) so binaries built on newer runners still run on older supported macOS versions.
  - Add a post-build verification step for macOS artifacts to check minimum OS version and architecture metadata before publishing.
  - Decide long-term packaging strategy for macOS (`arm64` + `x86_64` separate artifacts vs `universal2`) and align install scripts accordingly.
  - Add smoke tests on each produced artifact type (Linux, Windows, macOS Intel, macOS Apple Silicon) to confirm loadability/execution in target environments.
- Push new tag release 1.4
