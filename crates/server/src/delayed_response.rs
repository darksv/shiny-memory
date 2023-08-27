use std::future::Future;
use std::pin::{Pin, pin};
use std::task::Poll;
use std::time::Duration;

use axum::body::{Bytes, HttpBody};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use futures_util::future::BoxFuture;
use futures_util::FutureExt;

enum State<T> {
    WaitingForResult(BoxFuture<'static, T>),
    WaitingForResponse(Response),
}

pub(crate) struct DelayedResponse<T> {
    state: State<T>,
    ticker: tokio::time::Interval,
}

impl<T: Send + IntoResponse> DelayedResponse<T> {
    pub(crate) fn new(f: impl Future<Output=T> + Send + 'static) -> Self {
        Self {
            state: State::WaitingForResult(Box::pin(f)),
            ticker: tokio::time::interval(Duration::from_secs(20)),
        }
    }
}

impl<T: Sync + Send + 'static + IntoResponse> IntoResponse for DelayedResponse<T> {
    fn into_response(self) -> Response {
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
            match &mut self.state {
                State::WaitingForResult(fut) => {
                    if let Poll::Ready(result) = fut.poll_unpin(cx) {
                        self.state = State::WaitingForResponse(result.into_response());
                        // Immediately try to poll the response
                        continue;
                    }
                }
                State::WaitingForResponse(resp) => {
                    return pin!(resp).poll_data(cx);
                }
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
