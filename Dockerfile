FROM rust:1.68 AS builder

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/app/target \
		--mount=type=cache,target=/usr/local/cargo/git \
		--mount=type=cache,target=/usr/local/rustup \
		set -eux; \
        rustup set profile minimal; \
        rustup default nightly; \
	 	cargo build --release --bin server ; \
		objcopy --compress-debug-sections target/release/server ./server

FROM debian:11.3-slim

WORKDIR app

COPY --from=builder /app/server ./server
COPY --from=builder /app/best.onnx ./assets/models/best7.onnx

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates

RUN update-ca-certificates

ENV RUST_LOG=debug
ENV RUST_BACKTRACE=full
CMD ["./server"]