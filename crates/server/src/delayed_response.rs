use std::cell::Cell;
use std::future::Future;
use std::pin::{Pin};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::time::Duration;
use axum::body::{Bytes, HttpBody};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use futures_util::future::BoxFuture;

pub(crate) struct DelayedResponse<T> {
    generate: Option<BoxFuture<'static, T>>,
    tx: Option<tokio::sync::oneshot::Sender<Option<Result<Bytes, axum::Error>>>>,
    rx: tokio::sync::oneshot::Receiver<Option<Result<Bytes, axum::Error>>>,
    done: Arc<AtomicBool>,
    last_send: Cell<std::time::Instant>,
    interval: Duration,
}

impl<T: Send + IntoResponse> DelayedResponse<T> {
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

impl<T: Sync + Send + 'static + IntoResponse> IntoResponse for DelayedResponse<T> {
    fn into_response(self) -> axum::response::Response {
        self.boxed_unsync().into_response()
    }
}

impl<T> HttpBody for DelayedResponse<T>
    where T: IntoResponse + Send + 'static
{
    type Data = Bytes;
    type Error = axum::Error;

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
                let res = result.into_response().data().await;
                waker.wake();
                _ = tx.send(res);
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
            self.done.store(true, Ordering::Relaxed);
            Poll::Ready(data)
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
