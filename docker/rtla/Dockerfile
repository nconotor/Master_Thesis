FROM archlinux:latest 

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm git base-devel bc libelf trace-cmd python-docutils && \
    git clone --depth 1 --branch v6.6.28 https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git /linux-rtla && \
    cd /linux-rtla/tools/tracing/rtla && \
    make -j"$(nproc)" && \
    make install

ENTRYPOINT ["/usr/bin/rtla"]