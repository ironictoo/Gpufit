# TODO
## GPUfit / CPUfit Handoff Items (for external accelerator project)
Status update (2026-02-22) after ROCKETSHIP-driven upstream debugging:
- Completed in `Gpufit/Cpufit` (`dev`):
  - `TOFTS_EXTENDED` short-timer failure fixed in `Cpufit` (analytic derivative path in CPU solver; ROCKETSHIP handoff repro now returns `state=0` with finite parameters close to CPU reference).
  - Deterministic failed-fit handling improved in `Cpufit` (non-finite curve/derivative/chi-square checks now exit with an explicit fit state instead of silently propagating invalid math).
  - Constrained multi-fit bound handling bug fixed in `Cpufit` (`constraints` row now offsets per fit instead of reusing fit 0 bounds for all fits).
  - Python backend diagnostics helpers added in `pycpufit` and `pygpufit` (`fit_state_name`, `summarize_fit_states`) so downstream tests can print fit-state summaries directly.
  - `2CXM` forward-model formula bug fixed in both CPU and CUDA implementations (time-dependent exponential term was incorrect in one branch).
  - `tissue_uptake` numerical robustness improved (domain guards + adaptive finite-difference steps); ROCKETSHIP `test_osipi_pycpufit_tissue_uptake_fast` now passes locally (`XPASS` while still marked `xfail` downstream).
- `TOFTS_EXTENDED` qualification follow-up:
  - Remaining ROCKETSHIP voxel-batch `ex_tofts` failures were traced primarily to `MAX_ITERATION` under a caller-provided accelerated tolerance of `1e-12` (too strict for float32 Cpufit convergence checks).
  - Upstream guardrail kept in `Cpufit`: `TOFTS_EXTENDED` effective tolerance floor set to `1e-8` (small safety net only).
  - Recommended downstream production/test setting (ROCKETSHIP): set accelerated `ex_tofts` tolerance explicitly to `1e-6` (for example via model-specific `ex_tofts_gpu_tolerance`) so test/production behavior is intentional and visible in config.
  - With explicit practical tolerance downstream, ROCKETSHIP qualification `ex_tofts` finiteness recovered above the primary-model gate on BIDS test sessions.
- Repro / debugging assets retained:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/contracts/handoffs/cpufit_tofts_extended/`
  - Runner:
    - `.venv/bin/python tests/contracts/handoffs/cpufit_tofts_extended/run_cpufit_tofts_extended_repro.py --json`

Remaining handoff work (focus on `2cxm`):
- [ ] Improve constrained-fit robustness/accuracy for `2cxm` in `Cpufit` (current representative OSIPI fast case no longer silently returns an incorrect converged fit, but still fails to reach acceptable solution quality reliably).
- [ ] Investigate `2cxm` parameterization/conditioning in LM solver (`Ktrans/ve/vp/fP` bounds, initialization strategy, and derivative stability near `Ktrans -> fP` and boundary-clamped solutions).
- [ ] Add a dedicated `2cxm` handoff repro package (single-case + batch-case) similar to the `TOFTS_EXTENDED` handoff payload to speed upstream iteration and regression checks.
- [ ] Validate CUDA `2cxm` parity/behavior after CPU-side formula correction on a CUDA runner (CPU and CUDA source were both patched, but local validation here was CPU-only).

- Distribution compatibility hardening:
  - Set explicit macOS deployment targets for release builds (for example via `CMAKE_OSX_DEPLOYMENT_TARGET`) so binaries built on newer runners still run on older supported macOS versions.
  - Add a post-build verification step for macOS artifacts to check minimum OS version and architecture metadata before publishing.
  - Decide long-term packaging strategy for macOS (`arm64` + `x86_64` separate artifacts vs `universal2`) and align install scripts accordingly.
  - Add smoke tests on each produced artifact type (Linux, Windows, macOS Intel, macOS Apple Silicon) to confirm loadability/execution in target environments.
