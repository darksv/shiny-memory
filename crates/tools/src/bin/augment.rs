#![feature(try_blocks)]

use std::{fs, io};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::convert::identity;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Parser;
use image::{GenericImageView, ImageFormat};
use image::imageops::FilterType;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub annotations: Vec<TaskAnnotation>,
    pub file_upload: String,
    pub created_at: String,
    pub updated_at: String,
    pub inner_id: i64,
    pub total_annotations: i64,
    pub cancelled_annotations: i64,
    pub total_predictions: i64,
    pub comment_count: i64,
    pub project: i64,
    pub updated_by: i64,
    pub comment_authors: Vec<Rectangle>,
    pub data: Data,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Data {
    pub image: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskAnnotation {
    pub id: i64,
    pub completed_by: i64,
    pub result: Vec<Result>,
    pub was_cancelled: bool,
    pub ground_truth: bool,
    pub created_at: String,
    pub updated_at: String,
    pub lead_time: f64,
    pub result_count: i64,
    pub task: i64,
    pub project: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Result {
    pub original_width: i64,
    pub original_height: i64,
    pub image_rotation: i64,
    pub value: Rectangle,
    pub id: String,
    pub from_name: String,
    pub to_name: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub origin: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub rotation: f64,
    #[serde(default)]
    pub rectanglelabels: Vec<String>,
}

#[derive(Serialize)]
struct DataSet {
    path: String,
    train: String,
    val: String,
    nc: usize,
    names: Vec<String>,
}

fn dir_with_next_seq_number(base_path: impl AsRef<Path>) -> Option<(PathBuf, usize)> {
    let base_path = base_path.as_ref();
    let prefix = base_path.file_name()?.to_str()?;
    let parent_dir = base_path.parent()?;

    let seq = fs::read_dir(parent_dir)
        .unwrap()
        .flatten()
        .flat_map(|entry| Some(entry.path()
            .file_name()?.to_str()?
            .strip_prefix(prefix)?
            .parse::<usize>().ok()?))
        .max()
        .map_or(1, |max| max + 1);

    Some((parent_dir.join(&format!("{prefix}{seq}")), seq))
}

#[derive(Debug)]
struct InputFile {
    path: PathBuf,
    annotations: Vec<Annotation>,
    hash: u64,
    size: (u32, u32),
}

#[derive(clap::Parser)]
struct Config {
    #[arg(long, default_value = r"D:\ML\yolo\data")]
    data_dir: PathBuf,
    #[arg(long, default_value = r"C:\Users\Host\Downloads\project-1-at-2023-07-29-23-22-c7a8f372.json")]
    json_path: PathBuf,
    #[arg(long, default_value = r"D:\ML\label-studio\media")]
    label_studio_media_dir: PathBuf,
    #[arg(short, long)]
    dry_run: bool,
    #[arg(long, default_value = r"D:\ML\yolo\.cache")]
    cache_dir: PathBuf,
    #[arg(long, default_value = r"data.yaml")]
    data_yaml: PathBuf,
}

#[derive(Deserialize)]
struct DatasetConfig<'d> {
    #[serde(borrow)]
    classes: HashMap<&'d str, &'d str>,
}

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let (dataset_output_dir, n) = dir_with_next_seq_number(config.data_dir).unwrap();

    let yaml = fs::read(config.data_yaml).unwrap();
    let dataset_config: DatasetConfig = serde_yaml::from_slice(&yaml).unwrap();

    let file = fs::File::open(&config.json_path)?;
    let mut reader = io::BufReader::new(file);
    let mut data: Vec<Task> = serde_json::from_reader(&mut reader)?;
    for task in &mut data {
        if task.annotations.len() <= 1 {
            continue;
        }
        task.annotations.sort_by_key(|it| Reverse(it.id));
        task.annotations.drain(1..);
    }

    let classes: HashSet<_> = data.iter()
        .flat_map(|it| it.annotations.iter())
        .flat_map(|it| it.result.iter())
        .flat_map(|it| it.value.rectanglelabels.iter())
        .collect::<HashSet<_>>()
        .into_iter()
        .flat_map(|it| dataset_config.classes.get(it.as_str()))
        .collect();

    let classes = {
        let mut classes: Vec<_> = classes.into_iter().collect();
        classes.sort();
        classes
    };

    let mut base_images: Vec<_> = data.par_iter().filter_map(|task| {
        let rel_path = Path::new(&task.data.image);
        let path = rel_path.strip_prefix(r"\data").unwrap();
        let path = config.label_studio_media_dir.join(path);

        let image: image::ImageResult<_> = try {
            let reader = image::io::Reader::open(&path)?
                .with_guessed_format()?;
            let image = reader.decode()?;
            Some(image)
        };

        let image = match image {
            Ok(Some(image)) => image,
            Ok(None) => {
                println!("Error opening '{}': unknown format", path.display());
                return None;
            }
            Err(image::ImageError::IoError(e)) if e.kind() == io::ErrorKind::NotFound => return None,
            Err(e) => {
                println!("Error opening '{}': {:?}", path.display(), e);
                return None;
            }
        };

        let mut annotations = Vec::new();
        for ann in &task.annotations {
            if ann.was_cancelled {
                return None;
            }

            for res in &ann.result {
                for label in &res.value.rectanglelabels {
                    let Some(label) = dataset_config.classes.get(label.as_str()) else {
                        continue;
                    };

                    let w = res.value.width / 100.0;
                    let h = res.value.height / 100.0;
                    let x = res.value.x / 100.0;
                    let y = res.value.y / 100.0;
                    let cx = x + w / 2.0;
                    let cy = y + h / 2.0;

                    let label_id = classes.iter().position(|x| *x == label)
                        .expect("must exist");
                    annotations.push(Annotation { label_id, cx, cy, w, h });
                }
            }
        }

        let hasher = image_hasher::HasherConfig::new().to_hasher();
        let hash = hasher.hash_image(&image);
        let hash: [u8; 8] = hash.as_bytes().try_into()
            .expect("unsupported hash length");

        Some(InputFile {
            path,
            annotations,
            hash: u64::from_ne_bytes(hash),
            size: image.dimensions(),
        })
    }).collect();

    base_images.sort_by_key(|i| (i.hash, Reverse(i.size)));
    let before = base_images.len();
    base_images.dedup_by_key(|i| i.hash);
    let after = base_images.len();
    println!("Dedup: before={before}, after={after}");

    println!("Before augmentation");

    let mut counted = vec![0; classes.len()];
    for image in &base_images {
        for ann in &image.annotations {
            counted[ann.label_id] += 1;
        }
    }
    for (idx, count) in counted.iter().enumerate() {
        println!("{:<10} ({}): {}", classes[idx], idx, count);
    }

    let total: usize = counted.iter().copied().max().unwrap();

    // normalize to 0..=1.0
    let popularity_by_label: Vec<_> = counted
        .iter()
        .copied()
        .map(|it| it as f32 / total as f32)
        .collect();

    let mut transformations = [
        Transformation::None,
        Transformation::FlipHorizontal,
        Transformation::FlipVertical,
        Transformation::Rotate90,
        Transformation::Rotate270,
    ];

    let mut images = vec![];
    for (idx, image) in base_images.iter().enumerate() {
        // Images that contain most popular labels are less likely to be transformed:
        // each label has associated popularity factor based on individual contribution to the whole set.
        let popularity_ratio: f32 = image.annotations.iter()
            .map(|it| popularity_by_label[it.label_id])
            .sum::<f32>() / image.annotations.len() as f32;

        let transforms_to_create = (((1.0 - popularity_ratio) * 5.0).round() as usize)
            .clamp(1, 5);

        transformations.shuffle(&mut thread_rng());
        for op in transformations.iter().take(transforms_to_create) {
            images.push((idx, *op));
        }
    }

    println!("After augmentation");
    let mut counted = vec![0; classes.len()];
    for (idx, _) in &images {
        let image = &base_images[*idx];
        for ann in &image.annotations {
            counted[ann.label_id] += 1;
        }
    }
    for (idx, count) in counted.iter().enumerate() {
        println!("{:<10} ({}): {}", classes[idx], idx, count);
    }

    if config.dry_run {
        return Ok(());
    }

    fs::create_dir_all(&dataset_output_dir)?;

    let dataset_path = dataset_output_dir.join("data.yaml");
    let file = fs::File::options()
        .truncate(true)
        .create(true)
        .write(true)
        .open(dataset_path)
        .context("create yaml")?;

    let mut writer = io::BufWriter::new(file);
    serde_yaml::to_writer(&mut writer, &DataSet {
        path: format!("../data{n}/"),
        train: format!("images/training/"),
        val: format!("images/validation/"),
        nc: classes.len(),
        names: classes.iter().map(|it| it.to_string()).collect(),
    })?;
    drop(writer);

    images.shuffle(&mut thread_rng());

    let test_size = (0.1 * images.len() as f64).round() as usize;
    let (test, train) = images.split_at(test_size);

    fs::create_dir_all(&config.cache_dir).unwrap();

    for (dir, set) in [
        ("validation", test),
        ("training", train)
    ] {
        let labels = dataset_output_dir.join("labels").join(dir);
        let images = dataset_output_dir.join("images").join(dir);
        fs::create_dir_all(&labels).unwrap();
        fs::create_dir_all(&images).unwrap();

        set.into_par_iter().for_each(|(idx, op)| {
            let image = &base_images[*idx];

            let size = 640;
            let file_stem = image.path.file_stem().unwrap().to_str().unwrap();
            let file_name = match op {
                Transformation::None => format!("{}_{}", file_stem, size),
                op => format!("{}_{}_{:?}", file_stem, size, op),
            };

            let cached = config.cache_dir
                .join(&file_name)
                .with_extension("jpg");

            if !cached.exists() {
                let src = image::io::Reader::open(&image.path)
                    .unwrap()
                    .with_guessed_format()
                    .unwrap()
                    .decode()
                    .unwrap();
                let src = src.resize(size, size, FilterType::Lanczos3);
                let out = match op {
                    Transformation::FlipHorizontal => src.fliph(),
                    Transformation::FlipVertical => src.flipv(),
                    Transformation::Rotate90 => src.rotate90(),
                    Transformation::Rotate270 => src.rotate270(),
                    Transformation::None => src
                };
                out.save_with_format(&cached, ImageFormat::Jpeg).unwrap();
            }

            let image_out = images
                .join(&file_name)
                .with_extension("jpg");

            if !image_out.exists() {
                std::os::windows::fs::symlink_file(&cached, &image_out).unwrap();
            }

            let map_ann = match op {
                Transformation::FlipHorizontal => fliph,
                Transformation::FlipVertical => flipv,
                Transformation::Rotate90 => rotate90,
                Transformation::Rotate270 => rotate270,
                Transformation::None => identity,
            };

            let labels_out = labels
                .join(&file_name)
                .with_extension("txt");

            if !labels_out.exists() {
                let annotations: Vec<_> = image.annotations
                    .iter()
                    .map(|an| map_ann(*an))
                    .collect();
                save_labels(&labels_out, &annotations);
            }
        });
    }

    Ok(())
}

fn save_labels(path: impl AsRef<Path>, annotations: &[Annotation]) {
    if path.as_ref().exists() {
       return;
    }

    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();

    for ann in annotations {
        writeln!(&mut f, "{} {} {} {} {}", ann.label_id, ann.cx, ann.cy, ann.w, ann.h).unwrap();
    }
}

#[derive(Debug, Copy, Clone)]
enum Transformation {
    None,
    FlipHorizontal,
    FlipVertical,
    Rotate90,
    Rotate270,
}

#[derive(Default, Copy, Clone, Debug)]
struct Annotation {
    label_id: usize,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
}

fn fliph(ann: Annotation) -> Annotation {
    Annotation {
        cx: 1.0 - ann.cx,
        ..ann
    }
}

fn flipv(ann: Annotation) -> Annotation {
    Annotation {
        cy: 1.0 - ann.cy,
        ..ann
    }
}

fn rotate90(ann: Annotation) -> Annotation {
    Annotation {
        cx: 1.0 - ann.cy,
        cy: ann.cx,
        w: ann.h,
        h: ann.w,
        ..ann
    }
}

fn rotate270(ann: Annotation) -> Annotation {
    Annotation {
        cx: ann.cy,
        cy: 1.0 - ann.cx,
        w: ann.h,
        h: ann.w,
        ..ann
    }
}