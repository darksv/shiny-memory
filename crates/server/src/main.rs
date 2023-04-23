use std::fs::File;
use std::io::{BufReader, Cursor};
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::{http::StatusCode, Json, response::IntoResponse, Router, routing::{get, post}};
use axum::body::{Bytes, StreamBody};
use axum::extract::{Query, State};
use axum::http::header;
use futures_util::{Stream, StreamExt};
use image::{DynamicImage, GenericImage, GenericImageView, ImageFormat, Rgb};
use imageproc::rect::Rect;
use ort::AllocatorType;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::io::ReaderStream;
use tokio_util::sync::CancellationToken;

use crate::prediction::Detection;

mod prediction;
mod colors;
mod video;

#[derive(Clone)]
struct AppState {
    model: Arc<ort::Session>,
    model_version: u32,
    labels: Vec<String>,
    font: rusttype::Font<'static>,
}

fn parse_labels(s: &str) -> anyhow::Result<Vec<&str>> {
    let mut chars = s.chars();
    if chars.next() != Some('{') {
        anyhow::bail!("invalid start");
    }

    if chars.next_back() != Some('}') {
        anyhow::bail!("invalid end");
    }

    let mut labels = Vec::new();
    for item in chars.as_str().split(", ") {
        let (key, value) = item.split_once(": ")
            .context("not a 'key: value' pair")?;
        let index = key.parse::<usize>()?;
        let value = value.trim_matches('\'');
        labels.push((index, value));
    }

    Ok(labels.into_iter().map(|it| it.1).collect())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();

    let model_version = std::env::args()
        .nth(1)
        .context("missing model version")?
        .parse()
        .context("invalid model version")?;

    let path = format!(r"./assets/models/best{model_version}.onnx");
    tracing::info!("Loading model from '{}'", path);

    let environment = ort::Environment::builder()
        .with_name("test")
        .with_log_level(ort::LoggingLevel::Verbose)
        .build()?
        .into_arc();

    let session = ort::SessionBuilder::new(&environment)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level2)?
        .with_allocator(AllocatorType::Arena)?
        .with_intra_threads(1)?
        .with_model_from_file(&path)?;

    let labels = session
        .metadata()?
        .custom("names")?
        .context("missing property")?;

    let labels = parse_labels(&labels)?;
    tracing::info!("found labels: {:?}", labels);

    let font = include_bytes!("../../../assets/fonts/Roboto/Roboto-Regular.ttf");
    let font = rusttype::Font::try_from_bytes(font).expect("invalid font");

    video::init();

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(root))
        .route("/setup", post(setup))
        .route("/predict", post(predict_ls))
        .route("/infer", get(infer))
        .route("/draw", get(infer_draw))
        .route("/webhook", post(webhook))
        .with_state(AppState {
            model: Arc::new(session),
            model_version,
            labels: labels.iter().map(|l| l.to_string()).collect(),
            font,
        });

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn root() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({
        "status": "UP",
        "model_dir": "xx",
        "v2": true
    })))
}

async fn setup(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> (StatusCode, Json<Value>) {
    tracing::debug!("payload = {:?}", payload);
    (StatusCode::OK, Json(json!({
        "model_version": state.model_version
    })))
}

async fn webhook(
    Json(_payload): Json<Value>,
) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({})))
}

fn predicto(state: &AppState, payload: Predict) -> anyhow::Result<(Vec<Detection>, (u32, u32))> {
    let file = match &payload.tasks[0].data {
        TaskData::Image { image } => image,
    };

    let rel_path = file.strip_prefix("/data/").context("invalid path")?;
    let path = PathBuf::from(r"D:\label-studio\media")
        .join(rel_path)
        .canonicalize()?;

    tracing::info!("processing {}", path.display());

    let image = image::open(path)?.to_rgb8();
    let size = image.dimensions();
    let detections = prediction::predict(&state.model, &image)?;
    Ok((detections, size))
}

#[derive(Serialize)]
struct InferenceBox {
    frame: usize,
    #[serde(flatten)]
    rect: BoxRect,
    class_id: usize,
    confidence: f32,
}

#[derive(Serialize)]
struct InferenceResponse {
    boxes: Vec<InferenceBox>,
    model_version: u32,
}

#[derive(Deserialize)]
struct PredictQuery {
    url: String,
    #[serde(default)]
    labels: bool,
    #[serde(default)]
    single_frame: bool,
    #[serde(default)]
    step: Option<usize>,
}

enum Media {
    Image(DynamicImage),
    Video(PathBuf),
}

struct InferenceResult {
    detections: Vec<(usize, Detection)>,
    file: Media,
    size: (u32, u32),
}

async fn save_stream_to_file(
    input_file_path: &Path,
    mut stream: impl Stream<Item=reqwest::Result<Bytes>> + Unpin,
) -> anyhow::Result<()> {
    let mut input_file = tokio::fs::File::create(input_file_path).await?;
    while let Some(item) = stream.next().await {
        tokio::io::copy(&mut item?.as_ref(), &mut input_file).await?;
    }
    Ok(())
}

const PROCESSING_TIMEOUT: Duration = Duration::from_secs(45);

async fn infer_from_url(
    url: &str,
    state: &AppState,
    single_frame: bool,
    step: usize,
) -> anyhow::Result<InferenceResult> {
    tracing::info!("downloading... {url}");
    let response = reqwest::get(url).await?;
    let is_video = response.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|it| it.to_str().ok())
        .is_some_and(|it| it.starts_with("video/"));

    let input_file_path = Path::new("input.bin");
    save_stream_to_file(&input_file_path, response.bytes_stream()).await?;

    if is_video {
        let token_source = CancellationToken::new();
        let token = token_source.clone();
        let model = state.model.clone();

        let handle = tokio::task::spawn_blocking(move || {
            infer_video(input_file_path, InferVideoConfig { step, single_frame }, &model, token)
        });

        let _watch_handle = tokio::task::spawn(async move {
            tokio::time::sleep(PROCESSING_TIMEOUT).await;
            tracing::info!("timeout");
            token_source.cancel();
        });

        let infer_result = handle.await
            .context("task join error")?
            .context("video inference task")?;

        Ok(InferenceResult {
            detections: infer_result.detections,
            file: match infer_result.first_frame {
                Some(frame) => Media::Image(frame.into()),
                None => Media::Video(input_file_path.into()),
            },
            size: infer_result.frame_size.unwrap_or((0, 0)),
        })
    } else {
        let file = File::open(input_file_path)?;
        let buf = BufReader::new(file);
        let img = image::io::Reader::new(buf)
            .with_guessed_format()?
            .decode()?
            .into_rgb8();

        tracing::info!("inferring classes...");
        let detections = prediction::predict(&state.model, &img)?
            .into_iter()
            .map(|d| (0, d))
            .collect();
        tracing::info!("done.");

        Ok(InferenceResult {
            detections,
            size: img.dimensions(),
            file: Media::Image(img.into()),
        })
    }
}

struct InferVideoConfig {
    step: usize,
    single_frame: bool,
}

struct VideoInferenceResult {
    detections: Vec<(usize, Detection)>,
    first_frame: Option<image::RgbImage>,
    frame_size: Option<(u32, u32)>,
}

fn infer_video(
    input_path: &Path,
    config: InferVideoConfig,
    model: &ort::Session,
    token: CancellationToken,
) -> anyhow::Result<VideoInferenceResult> {
    let mut detections = Vec::new();
    let mut frame_size = None;
    let mut first_frame = None;

    tracing::info!("decoding video frames...");
    video::decode_video(input_path, |frame_idx, frame| {
        if token.is_cancelled() {
            tracing::warn!("cancelled decoder task");
            return ControlFlow::Break(());
        }

        if frame_idx % config.step == 0 {
            let result = match prediction::predict(&model, frame) {
                Ok(det) => det,
                Err(e) => {
                    tracing::warn!("error during prediction: {}", e);
                    return ControlFlow::Break(());
                }
            };
            for detection in result {
                detections.push((frame_idx, detection));
            }
        }

        frame_size = Some(frame.dimensions());

        if config.single_frame {
            let mut image = image::RgbImage::new(frame.width(), frame.height());
            image::imageops::overlay(&mut image, frame, 0, 0);
            first_frame = Some(image);
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }).context("decoding video")?;

    tracing::info!("done.");
    Ok(VideoInferenceResult { detections, first_frame, frame_size })
}

#[derive(Serialize)]
struct BoxRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn rect_to_box_rect(rect: Rect, real_width: u32, real_height: u32) -> BoxRect {
    let x = rect.left() as f32 / real_width as f32;
    let y = rect.top() as f32 / real_height as f32;
    let width = rect.width() as f32 / real_width as f32;
    let height = rect.height() as f32 / real_height as f32;

    let x = x.clamp(0.0, 1.0);
    let y = y.clamp(0.0, 1.0);
    let width = width.clamp(0.0, 1.0 - x);
    let height = height.clamp(0.0, 1.0 - y);
    BoxRect { x, y, width, height }
}

async fn infer(
    State(state): State<AppState>,
    Query(payload): Query<PredictQuery>,
) -> impl IntoResponse {
    match infer_from_url(&payload.url, &state, payload.single_frame, payload.step.unwrap_or(1)).await {
        Ok(InferenceResult { detections, file: _, size: (width, height) }) => {
            let mut boxes = vec![];
            for &(frame, ref detection) in &detections {
                boxes.push(InferenceBox {
                    frame,
                    rect: rect_to_box_rect(detection.rect, width, height),
                    class_id: detection.class_id,
                    confidence: detection.conf,
                });
            }

            (StatusCode::OK, Json(InferenceResponse {
                boxes,
                model_version: state.model_version,
            })).into_response()
        }
        Err(e) => {
            tracing::warn!("inference error: {}", e);
            (StatusCode::BAD_REQUEST, Json(json!({
                "error": e.to_string()
            }))).into_response()
        }
    }
}

fn draw_box(
    state: &AppState,
    output: &mut impl GenericImage<Pixel=Rgb<u8>>,
    detection: &Detection,
    show_labels: bool,
) {
    let color = colors::get_color(detection.class_id)
        .expect("too many classes?!");

    imageproc::drawing::draw_hollow_rect_mut(
        output,
        detection.rect,
        color,
    );

    let padding = 2;
    let text_scale = rusttype::Scale::uniform(10.0);
    let label = if show_labels {
        format!("{} — {:.02}", state.labels[detection.class_id], detection.conf)
    } else {
        format!("{} — {:.02}", detection.class_id, detection.conf)
    };
    let (w, h) = imageproc::drawing::text_size(text_scale, &state.font, &label);

    let text_bg = Rect::at(
        detection.rect.left(),
        detection.rect.top(),
    ).of_size(
        w as u32 + padding * 2,
        h as u32 + padding * 2,
    );
    imageproc::drawing::draw_filled_rect_mut(output, text_bg, color);
    imageproc::drawing::draw_text_mut(
        output,
        colors::optimal_text_color_for_background(color),
        detection.rect.left() + padding as i32,
        detection.rect.top() + padding as i32,
        text_scale,
        &state.font,
        &label,
    );
}

async fn infer_draw(
    State(state): State<AppState>,
    Query(payload): Query<PredictQuery>,
) -> impl IntoResponse {
    match infer_from_url(&payload.url, &state, payload.single_frame, payload.step.unwrap_or(1)).await {
        Ok(InferenceResult { detections, file: Media::Image(img), size: _ }) => {
            let mut output = img.to_rgb8();
            for (_, detection) in detections {
                draw_box(&state, &mut output, &detection, payload.labels);
            }

            let mut cursor = Cursor::new(vec![]);
            output.write_to(&mut cursor, ImageFormat::Png).unwrap();

            (
                [
                    (header::CONTENT_TYPE, "image/png"),
                    (header::CONTENT_DISPOSITION, "inline; filename=\"image.png\"")
                ],
                cursor.into_inner()
            ).into_response()
        }
        Ok(InferenceResult { detections, file: Media::Video(path), size: _ }) => {
            let mut det_iter = detections.into_iter();
            video::overlay_video(path, "output.mp4", move |idx, frame| {
                for (_, detection) in det_iter.by_ref().take_while(|(fidx, _)| idx == *fidx) {
                    draw_box(&state, frame, &detection, payload.labels);
                }
            }).unwrap();

            let file = tokio::fs::File::open("output.mp4").await.unwrap();
            let stream = ReaderStream::new(file);
            let body = StreamBody::new(stream);

            (
                [
                    (header::CONTENT_TYPE, "video/mp4"),
                    (header::CONTENT_DISPOSITION, "inline; filename=\"video.mp4\"")
                ],
                body
            ).into_response()
        }
        Err(e) => {
            tracing::warn!("inference error: {}", e);
            (StatusCode::BAD_REQUEST, Json(json!({
                "error": e.to_string()
            }))).into_response()
        }
    }
}

fn to_ls_pos(x: f32) -> f32 {
    (x * 100.0).clamp(0.0, 100.0)
}

async fn predict_ls(
    State(state): State<AppState>,
    Json(payload): Json<Predict>,
) -> (StatusCode, Json<PredictResults>) {
    let mut preds = vec![];
    let score = match predicto(&state, payload) {
        Ok((dets, (width, height))) => {
            let mut total_score = 0.0;
            for detection in &dets {
                preds.push(PredictResult {
                    original_width: width,
                    original_height: height,
                    image_rotation: 0,
                    r#type: "rectanglelabels".into(),
                    value: PredictValue {
                        x: to_ls_pos(detection.rect.left() as f32 / width as f32),
                        y: to_ls_pos(detection.rect.top() as f32 / height as f32),
                        width: to_ls_pos(detection.rect.width() as f32 / width as f32),
                        height: to_ls_pos(detection.rect.height() as f32 / height as f32),
                        rotation: 0.0,
                        rectanglelabels: vec![state.labels[detection.class_id].clone()],
                    },
                    score: detection.conf,
                    from_name: "label".to_string(),
                    to_name: "image".to_string(),
                });
                total_score += detection.conf;
            }

            if preds.is_empty() {
                0.0
            } else {
                total_score / (preds.len() as f32)
            }
        }
        Err(e) => {
            tracing::error!("err {}", e);
            0.0
        }
    };

    let res = PredictResults {
        results: vec![Item { result: preds, score }],
        model_version: state.model_version,
    };

    (StatusCode::OK, Json(res))
}


#[derive(Serialize, Debug)]
struct Item {
    result: Vec<PredictResult>,
    score: f32,
}

#[derive(Serialize, Debug)]
struct PredictResults {
    results: Vec<Item>,
    model_version: u32,
}

#[derive(Serialize, Debug)]
struct PredictResult {
    original_width: u32,
    original_height: u32,
    image_rotation: u32,
    value: PredictValue,
    r#type: String,
    from_name: String,
    to_name: String,
    score: f32,
}

#[derive(Serialize, Debug)]
struct PredictValue {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    rotation: f32,
    rectanglelabels: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct Predict {
    tasks: Vec<Task>,
}

#[derive(Deserialize, Debug)]
struct Task {
    data: TaskData,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum TaskData {
    Image { image: String }
}
