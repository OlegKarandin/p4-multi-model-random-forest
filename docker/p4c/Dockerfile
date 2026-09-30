# p4c for the compiler-verified campaign (spec 2026-09-29 §8.1): the WSL
# open-p4studio install, copied -- not rebuilt (a rebuild is 5 h+ and the last
# local build log ends in failure).
#
# Base choice: the same Debian-based miniconda image the devcontainer used
# before, so the Codespace conda setup is unchanged. If `ldd` in the image still
# reports `not found` after extra-libs, add the package to the apt-get line
# (build.sh step 5 is the check). If glibc proves too old for the WSL-built
# binaries, switch FROM to an Ubuntu 22.04 base (the spec's choice) and install
# miniconda in it. Base that worked: <record here after the first build>.
FROM mcr.microsoft.com/devcontainers/miniconda:3
ARG P4STUDIO_COMMIT
COPY install /opt/open-p4studio/install
COPY extra-libs/ /usr/local/lib/thesis-p4c/
RUN echo /usr/local/lib/thesis-p4c > /etc/ld.so.conf.d/thesis-p4c.conf && ldconfig \
 && apt-get update && apt-get install -y --no-install-recommends cpp libboost-all-dev \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/open-p4studio/install/bin:$PATH \
    SDE_INSTALL=/opt/open-p4studio/install \
    THESIS_P4STUDIO_COMMIT=$P4STUDIO_COMMIT
