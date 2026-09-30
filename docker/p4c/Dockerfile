# p4c for the compiler-verified campaign (spec 2026-09-29 §8.1): the WSL
# open-p4studio install, copied -- not rebuilt (a rebuild is 5 h+ and the last
# local build log ends in failure).
#
# Base choice: Ubuntu 22.04 (jammy), the OS the WSL install was built on, so the
# binaries see the same glibc 2.35 they were linked against. Miniforge supplies
# conda at /opt/conda, the path .devcontainer/postCreate.sh sources. (The
# devcontainers/miniconda images are all Debian 13 now, where the sshd feature's
# systemd/openssh-server install fails inside a container build -- measured
# 2026-09-30 in a Codespace creation log.) Miniforge is conda-forge only, which
# matches environment-linux.yml and avoids the Anaconda channel terms prompt.
# Base that worked: mcr.microsoft.com/devcontainers/base:jammy + Miniforge3.
FROM mcr.microsoft.com/devcontainers/base:jammy
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
 && chown -R vscode:vscode /opt/conda && ln -s /opt/conda/bin/conda /usr/local/bin/conda
ARG P4STUDIO_COMMIT
COPY install /opt/open-p4studio/install
COPY extra-libs/ /usr/local/lib/thesis-p4c/
RUN echo /usr/local/lib/thesis-p4c > /etc/ld.so.conf.d/thesis-p4c.conf && ldconfig \
 && apt-get update && apt-get install -y --no-install-recommends cpp libboost-all-dev \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/open-p4studio/install/bin:$PATH \
    SDE_INSTALL=/opt/open-p4studio/install \
    THESIS_P4STUDIO_COMMIT=$P4STUDIO_COMMIT
