FROM archlinux:latest

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
        gcc \
        make \
        numactl \
        util-linux \
        git && \
    git clone --branch v2.6 https://git.kernel.org/pub/scm/utils/rt-tests/rt-tests.git /opt/rt-tests && \
    cd /opt/rt-tests && \
    make && \
    make install && \
    pacman -Rs --noconfirm \
        git \
        make && \
    pacman -Scc --noconfirm && \
    rm -rf /opt/rt-tests

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /output

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
