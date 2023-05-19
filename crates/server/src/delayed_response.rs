use std::cell::Cell;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::time::Duration;
use axum::body::{Bytes, HttpBody};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use futures_util::future::BoxFuture;
use serde::Serialize;

pub(crate) struct DelayedResponse<T> {
    generate: Option<BoxFuture<'static, T>>,
    tx: Option<tokio::sync::oneshot::Sender<T>>,
    rx: tokio::sync::oneshot::Receiver<T>,
    done: Arc<AtomicBool>,
    last_send: Cell<std::time::Instant>,
    interval: Duration,
}

impl<T: Send> DelayedResponse<T> {
    pub(crate) fn new(f: impl Future<Output=T> + Send + 'static) -> Self {
        let (tx, rx) = tokio::sync::oneshot::channel();
        Self {
            generate: Some(Box::pin(f)),
            tx: Some(tx),
            rx,
            done: Arc::new(AtomicBool::new(false)),
            last_send: Cell::new(std::time::Instant::now()),
            interval: Duration::from_secs(5),
        }
    }
}

impl<T: Serialize + Sync + Send + 'static> IntoResponse for DelayedResponse<T> {
    fn into_response(self) -> axum::response::Response {
        self.boxed_unsync().into_response()
    }
}

impl<T: Send + 'static> HttpBody for DelayedResponse<T> where T: Serialize {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_data(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>> {
        if self.last_send.get().elapsed() >= self.interval {
            self.last_send.set(std::time::Instant::now());
            tracing::debug!("sending keep-alive...");
            return Poll::Ready(Some(Ok(Bytes::from("\n"))));
        }

        if let Some(fut) = self.generate.take() {
            let tx = self.tx.take().unwrap();
            let waker = cx.waker().clone();
            tokio::spawn(async move {
                let result = fut.await;
                waker.wake();
                _ = tx.send(result);
            });
            let waker = cx.waker().clone();
            let flag = self.done.clone();
            let mut interval = tokio::time::interval(self.interval);
            tokio::spawn(async move {
                loop {
                    if flag.load(Ordering::Relaxed) {
                        break;
                    }

                    interval.tick().await;
                    waker.wake_by_ref();
                }
            });

            Poll::Pending
        } else if let Ok(data) = self.rx.try_recv() {
            let serialized = serde_json::to_vec(&data).map_err(drop).unwrap();
            self.done.store(true, Ordering::Relaxed);
            Poll::Ready(Some(Ok(Bytes::from(serialized))))
        } else if self.done.load(Ordering::Relaxed) {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }

    fn poll_trailers(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<Option<HeaderMap>, Self::Error>> {
        Poll::Ready(Ok(None))
    }
}
