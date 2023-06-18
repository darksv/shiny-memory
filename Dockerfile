FROM rust:1.68-bullseye AS builder

WORKDIR /app
COPY . .

ENV ORT_LIB_LOCATION /usr/local/lib/libonnxruntime.so
ENV ORT_STRATEGY system
ENV RUSTFLAGS -C target-cpu=native

RUN ln -s /app/assets/libonnxruntime.so.1.14.1 /usr/local/lib/libonnxruntime.so
RUN ln -s /app/assets/libonnxruntime.so.1.14.1 /usr/local/lib/libonnxruntime.so.1.14.1

# TODO: clean it up
RUN --mount=type=cache,target=/app/target \
		--mount=type=cache,target=/usr/local/cargo/git \
		--mount=type=cache,target=/usr/local/rustup \
		set -eux; \
        apt-get update && \
        apt-get install -y clang libavcodec-dev libavformat-dev libavfilter-dev pkg-config \
        autoconf \
        automake \
        build-essential \
        cmake \
        git-core \
        libgnutls28-dev \
        libmp3lame-dev \
        libtool \
        libva-dev \
        libvdpau-dev \
        libvorbis-dev \
        libxcb1-dev \
        libxcb-shm0-dev \
        libxcb-xfixes0-dev \
        meson \
        ninja-build \
        pkg-config \
        texinfo \
        wget \
        yasm \
        zlib1g-dev; \
        rustup set profile minimal; \
        rustup update nightly; \
        rustup default nightly; \
	 	cargo build --release --bin server ; \
		objcopy --compress-debug-sections target/release/server ./server

FROM debian:11.6-slim

WORKDIR app

COPY --from=builder /app/server ./server
COPY --from=builder /app/assets ./assets

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates

RUN update-ca-certificates

ENV RUST_LOG="debug,server=INFO"
ENV RUST_BACKTRACE=full
ENV ORT_DYLIB_PATH /app/assets/libonnxruntime.so.1.14.1

CMD [
    "./server",
    "--model-version", "17",
    "--port", "3000",
    "--max-image-size-in-mb", "100"
]