use image::DynamicImage;
use image::imageops::FilterType;
use imageproc::rect::Rect;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_itertools::Itertools;
use tract_onnx::tract_hir::tract_ndarray::Axis;

const IMAGE_SIZE: u32 = 640;


pub(crate) type Model = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub(crate) fn predict(model: &Model, original_image: &DynamicImage) -> TractResult<Vec<Detection>> {
    let resized_image = original_image.resize(IMAGE_SIZE, IMAGE_SIZE, FilterType::Triangle);

    let mut input_image = image::RgbaImage::new(IMAGE_SIZE, IMAGE_SIZE);
    let horz_padding = IMAGE_SIZE - resized_image.width();
    let vert_padding = IMAGE_SIZE - resized_image.height();
    image::imageops::overlay(&mut input_image, &resized_image, (horz_padding / 2) as i64, (vert_padding / 2) as i64);

    let s = std::time::Instant::now();
    let input: Tensor = tract_ndarray::Array4::from_shape_fn(
        (1, 3, IMAGE_SIZE as _, IMAGE_SIZE as _),
        |(_, c, y, x)| {
            input_image.get_pixel(x as u32, y as u32).0[c] as f32 / 255.0
        }).into();
    tracing::debug!("{:?}", s.elapsed());

    let s = std::time::Instant::now();
    let result = model.run(tvec![input.into()])?;
    tracing::debug!("{:?}", s.elapsed());

    let predictions = result[0].to_array_view::<f32>()?;

    let longest = std::cmp::max(original_image.width(), original_image.height());
    let scale = longest as f32 / IMAGE_SIZE as f32;


    let mut boxes = Vec::new();
    for item in predictions.axis_iter(Axis(1)) {
        let Some(&[cx, cy, w, h, conf, ref classes @ ..]) = item.as_slice() else {
            unreachable!();
        };

        if conf <= 0.3 {
            continue;
        }

        let class_id = classes.iter().copied().position_max_by(f32::total_cmp).unwrap();
        let cx = (cx - horz_padding as f32 / 2.0) * scale;
        let cy = (cy - vert_padding as f32 / 2.0) * scale;
        let w = w * scale;
        let h = h * scale;

        let x = (cx - w / 2.0) as i32;
        let y = (cy - h / 2.0) as i32;

        let rect = Rect::at(x, y)
            .of_size(w as u32, h as u32);

        boxes.push(Detection { rect, conf, class_id });
    }

    let filtered = nms(&boxes, 0.1, 3);
    Ok(boxes.into_iter().enumerate().filter_map(|(idx, b)| filtered.contains(&idx).then_some(b)).collect())
}


fn nms(boxes: &[Detection], overlap_threshold: f32, neighbour_threshold: usize) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut indices = Vec::with_capacity(boxes.len());
    indices.extend(0..boxes.len());
    indices.sort_by_key(|&idx| (boxes[idx].rect.bottom(), idx));

    let mut remaining_boxes = Vec::new();

    while let Some(idx1) = indices.pop() {
        let box1 = &boxes[idx1];

        let mut neighbours = 0;
        indices.retain_mut(|idx2| {
            let box2 = &boxes[*idx2];
            if box1.class_id != box2.class_id {
                return true;
            }

            if box1.iou(&box2) > overlap_threshold {
                neighbours += 1;
                false
            } else {
                true
            }
        });

        if neighbours >= neighbour_threshold {
            remaining_boxes.push(idx1);
        }
    }

    remaining_boxes
}


#[derive(Debug)]
pub(crate) struct Detection {
    pub(crate) rect: Rect,
    pub(crate) conf: f32,
    pub(crate) class_id: usize,
}

impl Detection {
    fn area(&self) -> u32 {
        (self.rect.width() + 1) * (self.rect.height() + 1)
    }

    fn intersection_area(&self, other: &Self) -> u32 {
        self.rect.intersect(other.rect).map_or(0, |r| {
            (r.width() + 1) * (r.height() + 1)
        })
    }

    fn union_area(&self, other: &Self) -> u32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    fn iou(&self, other: &Self) -> f32 {
        let int_area = self.intersection_area(&other);
        let union_area = self.union_area(&other);
        let overlap = int_area as f32 / union_area as f32;
        overlap
    }
}
