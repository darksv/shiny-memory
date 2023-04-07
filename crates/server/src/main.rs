use std::fs;
use std::io::Cursor;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::Context;
use axum::{http::StatusCode, Json, response::IntoResponse, Router, routing::{get, post}};
use axum::body::StreamBody;
use axum::extract::{Query, State};
use axum::http::header;
use image::{DynamicImage, GenericImage, GenericImageView, ImageFormat, Rgb};
use imageproc::rect::Rect;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::io::ReaderStream;
use tracing_subscriber::EnvFilter;
use tract_onnx::onnx;
use tract_onnx::prelude::{Framework, InferenceModelExt};

use crate::prediction::{Detection, Model};

mod prediction;
mod colors;
mod video;

#[derive(Clone)]
struct AppState {
    model: Arc<Model>,
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
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_str("server=INFO").unwrap())
        .init();

    let model_version = 9;
    let path = format!(r"./assets/models/best{model_version}.onnx");

    let proto = onnx()
        .proto_model_for_path(&path)?;

    let labels = proto.metadata_props
        .into_iter()
        .find(|it| it.key == "names")
        .map(|it| it.value)
        .context("reading labels")?;

    let labels = parse_labels(&labels)?;
    tracing::info!("found labels: {:?}", labels);

    let model = onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;

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
            model: Arc::new(model),
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
    (StatusCode::OK, Json(json!({
        "model_version": state.model_version
    })))
}

async fn webhook(
    Json(_payload): Json<Value>,
) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({})))
}

fn predicto(state: &AppState, payload: Predict) -> anyhow::Result<(Vec<Detection>, DynamicImage)> {
    let file = match &payload.tasks[0].data {
        TaskData::Image { image } => image,
    };

    let rel_path = file.strip_prefix("/data/").context("invalid path")?;
    let path = PathBuf::from(r"D:\label-studio\media")
        .join(rel_path)
        .canonicalize()?;

    tracing::info!("processing {}", path.display());

    let image = image::open(path)?.to_rgb8();
    let detections = prediction::predict(&state.model, &image)?;
    Ok((detections, image.into()))
}

fn to_ls_pos(x: f32) -> f32 {
    (x * 100.0).clamp(0.0, 100.0)
}

#[derive(Serialize)]
struct InferenceBox {
    frame: usize,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
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

async fn infer_from_url(
    url: &str,
    state: &AppState,
) -> anyhow::Result<InferenceResult> {
    tracing::info!("downloading... {url}");
    let resp = reqwest::get(url).await?;
    match resp.headers().get(header::CONTENT_TYPE).and_then(|it| it.to_str().ok()) {
        Some(content_type) if content_type.starts_with("video/") => {
            let data = resp.bytes().await?;
            let path = "input.mp4";

            fs::write(path, data).context("saving output")?;
            let mut detections = Vec::new();
            let mut frame_size = None;

            tracing::info!("decoding video frames...");
            video::decode_video(path, |idx, frame| {
                let result = match prediction::predict(&state.model, frame) {
                    Ok(det) => det,
                    Err(e) => {
                        tracing::warn!("error during prediction: {}", e);
                        return;
                    }
                };

                for detection in result {
                    detections.push((idx, detection));
                }
                frame_size = Some(frame.dimensions());
            }).context("decoding video")?;
            tracing::info!("done.");

            Ok(InferenceResult {
                detections,
                file: Media::Video(path.into()),
                size: frame_size.unwrap_or((0, 0)),
            })
        }
        _ => {
            let data = resp.bytes().await?;
            tracing::info!("reading image...");
            let img = image::load_from_memory(&data)?.to_rgb8();
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
}

async fn infer(
    State(state): State<AppState>,
    Query(payload): Query<PredictQuery>,
) -> impl IntoResponse {
    match infer_from_url(&payload.url, &state).await {
        Ok(InferenceResult { detections, file: _, size: (width, height) }) => {
            let mut boxes = vec![];
            for &(frame, ref detection) in &detections {
                boxes.push(InferenceBox {
                    frame,
                    x: to_ls_pos(detection.rect.left() as f32 / width as f32),
                    y: to_ls_pos(detection.rect.top() as f32 / height as f32),
                    width: to_ls_pos(detection.rect.width() as f32 / width as f32),
                    height: to_ls_pos(detection.rect.height() as f32 / height as f32),
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
            tracing::error!("err {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, ()).into_response()
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
    match infer_from_url(&payload.url, &state).await {
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
            tracing::error!("inference error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, ()).into_response()
        }
    }
}

async fn predict_ls(
    State(state): State<AppState>,
    Json(payload): Json<Predict>,
) -> (StatusCode, Json<PredictResults>) {
    let mut preds = vec![];
    let score = match predicto(&state, payload) {
        Ok((dets, img)) => {
            let mut total_score = 0.0;
            for detection in &dets {
                preds.push(PredictResult {
                    original_width: img.width(),
                    original_height: img.height(),
                    image_rotation: 0,
                    r#type: "rectanglelabels".into(),
                    value: PredictValue {
                        x: to_ls_pos(detection.rect.left() as f32 / img.width() as f32),
                        y: to_ls_pos(detection.rect.top() as f32 / img.height() as f32),
                        width: to_ls_pos(detection.rect.width() as f32 / img.width() as f32),
                        height: to_ls_pos(detection.rect.height() as f32 / img.height() as f32),
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
