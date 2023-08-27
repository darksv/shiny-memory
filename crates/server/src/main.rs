#![feature(array_chunks)]

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Context;
use axum::{http::StatusCode, Json, response::IntoResponse, Router, routing::{get, post}};
use axum::body::{Bytes, StreamBody};
use axum::extract::{Query, State};
use axum::http::header;
use clap::Parser;
use futures_util::{Stream, StreamExt};
use gif::ColorOutput;
use image::{DynamicImage, GenericImage, GenericImageView, ImageFormat, Rgb};
use imageproc::rect::Rect;
use ort::AllocatorType;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::io::ReaderStream;
use tokio_util::sync::CancellationToken;

use crate::delayed_response::DelayedResponse;
use crate::prediction::Detection;
use crate::utils::TempFilePath;

mod prediction;
mod colors;
mod video;
mod delayed_response;
mod utils;

#[derive(Clone)]
struct AppState {
    models: Arc<RwLock<BTreeMap<u32, Arc<Model>>>>,
    max_image_size_in_bytes: u32,
    default_model_version: u32,
    font: rusttype::Font<'static>,
    environment: Arc<ort::Environment>,
    class_to_label_studio: HashMap<String, String>,
}

impl AppState {
    fn get_model(&self, version: Option<u32>) -> anyhow::Result<Arc<Model>> {
        let version = version.unwrap_or(self.default_model_version);
        if let Some(model) = self.models.read().unwrap().get(&version) {
            return Ok(model.clone());
        }

        let path = format!(r"./assets/models/best{version}.onnx");
        let model: Arc<_> = load_model(&self.environment, &path)?.into();
        self.models.write().unwrap().insert(version, model.clone());
        Ok(model)
    }
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

#[derive(clap::Parser)]
struct Config {
    #[arg(long)]
    model_version: u32,
    #[arg(long, default_value_t = 3000)]
    port: u16,
    #[arg(long, default_value_t = 100)]
    max_image_size_in_mb: u32,
    #[arg(long)]
    label_studio_yaml_path: Option<PathBuf>,
}

struct Model {
    session: ort::Session,
    labels: Vec<String>,
}

fn load_model(env: &Arc<ort::Environment>, path: impl AsRef<Path>) -> anyhow::Result<Model> {
    tracing::info!("Loading model from '{}'", path.as_ref().display());

    let session = ort::SessionBuilder::new(env)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level2)?
        .with_allocator(AllocatorType::Arena)?
        .with_parallel_execution(true)?
        .with_model_from_file(&path)?;

    let labels = session
        .metadata()?
        .custom("names")?
        .context("missing property")?;

    let labels = parse_labels(&labels)?;
    tracing::info!("found labels: {:?}", labels);

    Ok(Model {
        session,
        labels: labels.iter().map(|l| l.to_string()).collect(),
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Config::parse();
    tracing_subscriber::fmt().init();

    let dataset_config = if let Some(path) = args.label_studio_yaml_path {
        let yaml = fs::read(path).unwrap();
        serde_yaml::from_slice(&yaml).unwrap()
    } else {
        DatasetConfig::default()
    };

    let environment = ort::Environment::builder()
        .with_name("test")
        .with_log_level(ort::LoggingLevel::Verbose)
        .build()?
        .into_arc();

    let font = include_bytes!("../../../assets/fonts/Roboto/Roboto-Regular.ttf");
    let font = rusttype::Font::try_from_bytes(font).expect("invalid font");

    video::init();

    let state = AppState {
        environment,
        models: Default::default(),
        max_image_size_in_bytes: args.max_image_size_in_mb * 1024 * 1024,
        default_model_version: args.model_version,
        font,
        class_to_label_studio: dataset_config.class_to_label_studio,
    };

    // Try load default model
    let _model = state.get_model(None);

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(root))
        .route("/setup", post(setup))
        .route("/predict", post(predict_ls))
        .route("/infer", get(infer))
        .route("/model", get(model_info))
        .route("/draw", get(infer_draw))
        .route("/webhook", post(webhook))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
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
        "model_version": state.default_model_version
    })))
}

async fn webhook(
    Json(_payload): Json<Value>,
) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({})))
}

fn predicto(model: &Model, payload: Predict) -> anyhow::Result<(Vec<Detection>, (u32, u32))> {
    let file = match &payload.tasks[0].data {
        TaskData::Image { image } => image,
    };

    let rel_path = file.strip_prefix("/data/").context("invalid path")?;
    let path = PathBuf::from(r"D:\ML\label-studio\media")
        .join(rel_path)
        .canonicalize()?;

    tracing::info!("processing {}", path.display());

    let image = image::open(path)?.to_rgb8();
    let size = image.dimensions();
    let detections = prediction::predict(&model.session, &image, 0)?;
    Ok((detections, size))
}

#[derive(Serialize)]
struct InferenceBox {
    frame: usize,
    #[serde(flatten)]
    rect: BoxRect,
    class_id: usize,
    class_name: Option<String>,
    confidence: f32,
}

#[derive(Serialize)]
struct InferenceResponse {
    boxes: Vec<InferenceBox>,
    model_version: u32,
    error: Option<String>,
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
    #[serde(default)]
    timeout: bool,
    #[serde(default)]
    version: Option<u32>,
}

enum Media {
    Image(DynamicImage),
    Video(TempFilePath),
    Unknown,
}

struct InferenceResult {
    detections: Vec<(usize, Detection)>,
    media: Media,
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
    model: Arc<Model>,
    state: &AppState,
    single_frame: bool,
    step: usize,
    timeout: bool,
) -> anyhow::Result<InferenceResult> {
    tracing::info!("downloading... {url}");
    let response = reqwest::get(url).await?;
    let mime_type = response.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|it| it.to_str().ok())
        .map(|it| it.to_string());

    let input_path = TempFilePath::new("tmp", "bin")?;
    tracing::info!("Saving as {}", input_path.display());
    save_stream_to_file(&input_path, response.bytes_stream()).await?;

    let token_source = CancellationToken::new();
    let token = token_source.clone();
    let max_image_size_in_bytes = state.max_image_size_in_bytes;

    let handle = tokio::task::spawn_blocking(move || {
        match mime_type.as_deref() {
            Some("image/gif") => infer_gif(input_path, &model.session, token),
            Some(v) if v.starts_with("image/") => infer_image(input_path, &model.session, max_image_size_in_bytes),
            _ => infer_video(input_path, InferVideoConfig { step, single_frame }, &model.session, token),
        }
    });

    if timeout {
        let token = token_source.clone();
        tokio::task::spawn(async move {
            tokio::time::sleep(PROCESSING_TIMEOUT).await;
            token.cancel();
        });
    }

    let result = handle.await;
    if token_source.is_cancelled() {
        tracing::warn!("cancelled due to timeout");
    }

    result.context("task join error")?
}

fn infer_gif(
    input_path: TempFilePath,
    model: &ort::Session,
    token: CancellationToken,
) -> anyhow::Result<InferenceResult> {
    let file = File::open(&input_path)?;
    let reader = BufReader::new(file);

    let mut options = gif::DecodeOptions::new();
    options.set_color_output(ColorOutput::Indexed);
    let mut decoder = options.read_info(reader)?;
    let mut buf = Vec::new();

    let mut detections = Vec::new();
    let mut frame_idx = 0;
    let mut size = (0, 0);

    let mut screen = gif_dispose::Screen::new_decoder(&decoder);

    while let Some(frame) = decoder.read_next_frame()? {
        if token.is_cancelled() {
            break;
        }

        screen.blit_frame(&frame)?;

        buf.clear();
        buf.reserve(usize::from(frame.width) * usize::from(frame.height) * 3);
        buf.extend(screen.pixels.as_ref().pixels().flat_map(|p| [p.r, p.g, p.b]));

        let layout = image::flat::SampleLayout {
            channels: 3,
            channel_stride: 1,
            width: frame.width.into(),
            width_stride: 3,
            height: frame.height.into(),
            height_stride: usize::from(frame.width) * 3,
        };
        let samples = image::flat::FlatSamples {
            samples: &buf,
            layout,
            color_hint: None,
        };
        let image = samples.as_view()?;
        for detection in prediction::predict(model, &image, frame_idx)? {
            detections.push((frame_idx, detection));
        }
        size = (frame.width as u32, frame.height as u32);
        frame_idx += 1;
    }

    Ok(InferenceResult {
        detections,
        media: Media::Unknown,
        size,
    })
}

fn infer_image(
    input_path: TempFilePath,
    model: &ort::Session,
    max_image_size_in_bytes: u32,
) -> anyhow::Result<InferenceResult> {
    let file = File::open(&input_path)?;
    let (width, height) = image::io::Reader::new(BufReader::new(file))
        .with_guessed_format()?
        .into_dimensions()?;

    let raw_size_in_bytes = width * height * 3;
    tracing::info!("image dimensions = {width}x{height} = {raw_size_in_bytes}B");
    if raw_size_in_bytes >= max_image_size_in_bytes {
        anyhow::bail!("image is too big: {width}x{height}");
    }

    let file = File::open(&input_path)?;
    let image = image::io::Reader::new(BufReader::new(file))
        .with_guessed_format()?
        .decode()?
        .into_rgb8();

    tracing::info!("inferring classes...");
    let detections = prediction::predict(model, &image, 0)?
        .into_iter()
        .map(|d| (0, d))
        .collect();
    tracing::info!("done.");

    Ok(InferenceResult {
        detections,
        size: image.dimensions(),
        media: Media::Image(image.into()),
    })
}

struct InferVideoConfig {
    step: usize,
    single_frame: bool,
}

fn infer_video(
    input_path: TempFilePath,
    config: InferVideoConfig,
    model: &ort::Session,
    token: CancellationToken,
) -> anyhow::Result<InferenceResult> {
    let mut detections = Vec::new();
    let mut frame_size = None;
    let mut first_frame = None;

    tracing::info!("decoding video frames...");
    video::decode_video(&input_path, |frame_idx, frame| {
        if token.is_cancelled() {
            tracing::warn!("cancelled decoder task");
            return ControlFlow::Break(());
        }

        if frame_idx % config.step == 0 {
            let result = match prediction::predict(&model, frame, frame_idx) {
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

    Ok(InferenceResult {
        detections,
        media: match first_frame {
            Some(frame) => Media::Image(frame.into()),
            None => Media::Video(input_path),
        },
        size: frame_size.unwrap_or((0, 0)),
    })
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

#[derive(Serialize)]
struct ModelInfoResponse {
    version: u32,
    classes: Vec<String>,
}

async fn model_info(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let model = state.get_model(None).unwrap();
    ([(header::CONTENT_TYPE, "application/json")], Json(ModelInfoResponse {
        version: state.default_model_version,
        classes: model.labels.clone(),
    }))
}

async fn infer(
    State(state): State<AppState>,
    Query(payload): Query<PredictQuery>,
) -> impl IntoResponse {
    let model = state.get_model(payload.version).unwrap();
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], DelayedResponse::new(async move {
        match infer_from_url(&payload.url, model.clone(), &state, payload.single_frame, payload.step.unwrap_or(1), payload.timeout).await {
            Ok(InferenceResult { detections, media: _, size: (width, height) }) => {
                let mut boxes = vec![];
                for &(frame, ref detection) in &detections {
                    boxes.push(InferenceBox {
                        frame,
                        rect: rect_to_box_rect(detection.rect, width, height),
                        class_id: detection.class_id,
                        class_name: model.labels.get(detection.class_id).cloned(),
                        confidence: detection.conf,
                    });
                }

                Json(InferenceResponse {
                    boxes,
                    model_version: state.default_model_version,
                    error: None,
                })
            }
            Err(e) => {
                Json(InferenceResponse {
                    boxes: Vec::new(),
                    model_version: state.default_model_version,
                    error: Some(e.to_string()),
                })
            }
        }
    }))
}

fn draw_box(
    state: &AppState,
    model: &Model,
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
        format!("{} — {:.02}", model.labels[detection.class_id], detection.conf)
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
    let model = match state.get_model(payload.version) {
        Ok(model) => model,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({
            "error": e.to_string()
        }))).into_response()
    };

    match infer_from_url(&payload.url, model.clone(), &state, payload.single_frame, payload.step.unwrap_or(1), true).await {
        Ok(InferenceResult { detections, media: Media::Image(img), size: _ }) => {
            let mut output = img.to_rgb8();
            for (_, detection) in detections {
                draw_box(&state, &model, &mut output, &detection, payload.labels);
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
        Ok(InferenceResult { detections, media: Media::Video(path), size: _ }) => {
            let output_file = TempFilePath::new("tmp", "mp4").unwrap();
            let mut det_iter = detections.into_iter();
            video::overlay_video(&path, &output_file, move |idx, frame| {
                for (_, detection) in det_iter.by_ref().take_while(|(fidx, _)| idx == *fidx) {
                    draw_box(&state, &model, frame, &detection, payload.labels);
                }
            }).unwrap();

            let file = tokio::fs::File::open(&output_file).await.unwrap();
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
        Ok(InferenceResult { detections: _, media: Media::Unknown, size: _, }) => {
            unimplemented!()
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

#[derive(Deserialize, Default)]
struct DatasetConfig {
    class_to_label_studio: HashMap<String, String>,
}

async fn predict_ls(
    State(state): State<AppState>,
    Json(payload): Json<Predict>,
) -> (StatusCode, Json<PredictResults>) {
    let model = state.get_model(None).unwrap();

    let mut preds = vec![];
    let score = match predicto(&model, payload) {
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
                        rectanglelabels: {
                            let class_name = &model.labels[detection.class_id];
                            vec![
                                state.class_to_label_studio.get(class_name)
                                    .as_deref()
                                    .unwrap_or(class_name)
                                    .to_string()
                            ]
                        },
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
        model_version: state.default_model_version,
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
