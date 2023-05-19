use std::cell::Cell;
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
use tokio::sync::oneshot;

type Item = Option<Result<Bytes, axum::Error>>;

pub(crate) struct DelayedResponse<T> {
    future_with_tx: Option<(BoxFuture<'static, T>, oneshot::Sender<Item>)>,
    rx: oneshot::Receiver<Item>,
    done: Arc<AtomicBool>,
    last_send: Cell<std::time::Instant>,
    interval: Duration,
}

impl<T: Send + IntoResponse> DelayedResponse<T> {
    pub(crate) fn new(f: impl Future<Output=T> + Send + 'static) -> Self {
        let (tx, rx) = oneshot::channel();
        Self {
            future_with_tx: Some((Box::pin(f), tx)),
            rx,
            done: Arc::new(AtomicBool::new(false)),
            last_send: Cell::new(std::time::Instant::now()),
            interval: Duration::from_secs(20),
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
            return Poll::Ready(Some(Ok(Bytes::from("\n"))));
        }

        if let Some((future, tx)) = self.future_with_tx.take() {
            let waker = cx.waker().clone();
            let response_task = async move {
                let body = future.await.into_response().data().await;
                waker.wake();
                _ = tx.send(body);
            };
            let waker = cx.waker().clone();
            let flag = self.done.clone();
            let interval = self.interval;
            let ticker_task = async move {
                let mut interval = tokio::time::interval(interval);
                while !flag.load(Ordering::Relaxed) {
                    interval.tick().await;
                    waker.wake_by_ref();
                }
            };

            tokio::spawn(async move {
                tokio::join!(response_task, ticker_task)
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
