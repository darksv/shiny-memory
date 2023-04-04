use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use image::ImageFormat;
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
    pub rectanglelabels: Vec<String>,
}

#[derive(Serialize)]
struct DataSet {
    train: String,
    val: String,
    nc: usize,
    names: Vec<String>,
}

fn remap_label(label: &str) -> Option<&str> {
    Some(match label {

        _ => return None,
    })
}

fn main() -> anyhow::Result<()> {
    let n = 12;
    let output = PathBuf::from(format!(r"D:\yolo\data{n}"));
    let project_path = r"C:\Users\Host\Downloads\project-1-at-2023-04-03-14-36-67286744.json";

    let json = fs::read_to_string(project_path)?;
    let data: Vec<Task> = serde_json::from_str(&json)?;

    let classes: HashSet<_> = data.iter()
        .flat_map(|it| it.annotations.iter())
        .flat_map(|it| it.result.iter())
        .flat_map(|it| it.value.rectanglelabels.iter())
        .collect::<HashSet<_>>()
        .into_iter()
        .flat_map(|it| remap_label(it))
        .collect();

    let classes = {
        let mut classes: Vec<_> = classes.into_iter().collect();
        classes.sort();
        classes
    };

    let base = Path::new(r"D:\label-studio\media");
    let base_images: Vec<_> = data.par_iter().filter_map(|task| {
        let rel_path = Path::new(&task.data.image);
        let path = rel_path.strip_prefix(r"\data").unwrap();
        let path = base.join(path);
        if !image::open(&path).is_ok() {
            return None;
        }

        let mut annotations = Vec::new();
        for ann in &task.annotations {
            for res in &ann.result {
                for label in &res.value.rectanglelabels {
                    let Some(label) = remap_label(label) else {
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

        if annotations.is_empty() {
            return None;
        }

        Some((path, annotations))
    }).collect();

    println!("Before augmentation");

    let mut counted = vec![0; classes.len()];
    for (_, anns) in &base_images {
        for ann in anns {
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
    for (idx, (_, anns)) in base_images.iter().enumerate() {
        // Images that contain most popular labels are less likely to be transformed:
        // each label has associated popularity factor based on individual contribution to the whole set.
        let popularity_ratio: f32 = anns.iter()
            .map(|it| popularity_by_label[it.label_id])
            .sum::<f32>() / anns.len() as f32;

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
        let (_, anns) = &base_images[*idx];
        for ann in anns {
            counted[ann.label_id] += 1;
        }
    }
    for (idx, count) in counted.iter().enumerate() {
        println!("{:<10} ({}): {}", classes[idx], idx, count);
    }

    // return Ok(());


    fs::create_dir_all(&output)?;

    let dataset_path = output.join("data.yaml");
    let file = fs::File::options()
        .truncate(true)
        .create(true)
        .write(true)
        .open(dataset_path)
        .context("create yaml")?;

    let mut writer = BufWriter::new(file);
    serde_yaml::to_writer(&mut writer, &DataSet {
        train: format!("../data{n}/images/training/"),
        val: format!("../data{n}/images/validation/"),
        nc: classes.len(),
        names: classes.iter().map(|it| it.to_string()).collect(),
    })?;
    drop(writer);


    images.shuffle(&mut thread_rng());


    let test_size = (0.1 * images.len() as f64).round() as usize;
    let (test, train) = images.split_at(test_size);

    for (dir, set) in [
        ("validation", test),
        ("training", train)
    ] {
        let labels = output.join("labels").join(dir);
        let images = output.join("images").join(dir);
        fs::create_dir_all(&labels).unwrap();
        fs::create_dir_all(&images).unwrap();

        set.into_par_iter().for_each(|(idx, op)| {
            let (path, anns) = &base_images[*idx];

            let img = image::open(path).unwrap();
            let src = img.clone();
            let (out, map_ann): (_, fn(Annotation) -> Annotation) = match op {
                Transformation::FlipHorizontal => (src.fliph(), fliph),
                Transformation::FlipVertical => (src.flipv(), flipv),
                Transformation::Rotate90 => (src.rotate90(), rotate90),
                Transformation::Rotate270 => (src.rotate270(), rotate270),
                Transformation::None => (src, |x| x)
            };

            let anns: Vec<_> = anns.iter().map(|an| map_ann(*an)).collect();

            let file_stem = path.file_stem().unwrap().to_str().unwrap();
            let file_name = match op {
                Transformation::None => format!("{}", file_stem),
                op => format!("{}_{:?}", file_stem, op),
            };

            let image_out = images
                .join(&file_name)
                .with_extension("jpg");

            let labels_out = labels
                .join(&file_name)
                .with_extension("txt");

            if !image_out.exists() {
                out.save_with_format(&image_out, ImageFormat::Jpeg).unwrap();
            }

            if !labels_out.exists() {
                save_labels(&labels_out, &anns);
            }
        });
    }

    Ok(())
}

fn save_labels(path: impl AsRef<Path>, anns: &[Annotation]) {
    if !path.as_ref().exists() {
        let mut f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();

        for ann in anns {
            writeln!(&mut f, "{} {} {} {} {}", ann.label_id, ann.cx, ann.cy, ann.w, ann.h).unwrap();
        }
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