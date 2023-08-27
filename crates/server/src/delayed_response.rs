use std::future::Future;
use std::pin::{Pin, pin};
use std::task::Poll;
use std::time::Duration;

use axum::body::{Bytes, HttpBody};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use futures_util::future::BoxFuture;
use futures_util::FutureExt;

pub(crate) struct DelayedResponse<T> {
    future: BoxFuture<'static, T>,
    ticker: tokio::time::Interval,
    response: Option<Response>,
}

impl<T: Send + IntoResponse> DelayedResponse<T> {
    pub(crate) fn new(f: impl Future<Output=T> + Send + 'static) -> Self {
        Self {
            future: Box::pin(f),
            ticker: tokio::time::interval(Duration::from_secs(20)),
            response: None,
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
        loop {
            if let Some(response) = self.response.as_mut() {
                return pin!(response).poll_data(cx);
            }

            if let Poll::Ready(result) = self.future.poll_unpin(cx) {
                self.response = Some(result.into_response());
                continue;
            }

            if self.ticker.poll_tick(cx).is_ready() {
                return Poll::Ready(Some(Ok(Bytes::from("\n"))));
            }

            return Poll::Pending;
        }
    }

    fn poll_trailers(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<Option<HeaderMap>, Self::Error>> {
        Poll::Ready(Ok(None))
    }
}
