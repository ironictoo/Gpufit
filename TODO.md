# TODO

- Investigate intermittent/long Windows CI build stalls in `Build (Windows)` after Gradle reports Java build success; likely related to `JAVA_PACKAGE` in `Gpufit/java/CMakeLists.txt` being part of `ALL_BUILD`.
- Temporary mitigation is now enabled in both `.github/workflows/ci.yml` and `.github/workflows/release.yml`: Windows CMake configure disables Java/JNI package discovery (`CMAKE_DISABLE_FIND_PACKAGE_Java=TRUE`, `CMAKE_DISABLE_FIND_PACKAGE_JNI=TRUE`) and Java setup is skipped on Windows. Revert this after root-cause fix so full Java packaging coverage returns on Windows CI.
