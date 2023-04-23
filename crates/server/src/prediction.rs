use std::num::NonZeroU32;

use fast_image_resize as fr;
use image::{GenericImageView, Pixel, Rgb};
use image::flat::SampleLayout;
use imageproc::rect::Rect;
use ndarray::Axis;
use ort::OrtResult;
use ort::tensor::InputTensor;

const IMAGE_SIZE: u32 = 640;

pub(crate) trait GenericImageWithContinuousBuffer<P: Pixel>: GenericImageView<Pixel=P> {
    fn as_buffer(&self) -> &[P::Subpixel];
}

impl GenericImageWithContinuousBuffer<Rgb<u8>> for image::RgbImage {
    fn as_buffer(&self) -> &[u8] {
        self.as_ref()
    }
}

impl GenericImageWithContinuousBuffer<Rgb<u8>> for image::flat::View<&[u8], Rgb<u8>> {
    fn as_buffer(&self) -> &[u8] {
        self.image_slice()
    }
}

struct ResizingInfo {
    image: image::RgbImage,
    horz_padding: u32,
    vert_padding: u32,
}

fn resize(img: &impl GenericImageWithContinuousBuffer<Rgb<u8>>) -> ResizingInfo {
    let (new_width, new_height) = new_size_to_fit_preserving_aspect_ratio(img, IMAGE_SIZE, IMAGE_SIZE);

    let width = NonZeroU32::new(img.width()).unwrap();
    let height = NonZeroU32::new(img.height()).unwrap();

    let src_view: fr::ImageView<fr::pixels::U8x3> = fr::ImageView::from_buffer(
        width, height,
        img.as_buffer(),
    ).expect("image with valid size");

    let dst_width = NonZeroU32::new(new_width).unwrap();
    let dst_height = NonZeroU32::new(new_height).unwrap();

    let mut dst_image = fr::Image::new(
        dst_width,
        dst_height,
        src_view.pixel_type(),
    );

    let mut dst_view = dst_image.view_mut();
    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer.resize(&fr::DynamicImageView::from(src_view), &mut dst_view).unwrap();
    let resized_image = image::FlatSamples {
        samples: dst_image.buffer(),
        layout: SampleLayout {
            channels: 3,
            channel_stride: 1,
            width: dst_image.width().get(),
            width_stride: 3,
            height: dst_image.height().get(),
            height_stride: dst_image.width().get() as usize * 3,
        },
        color_hint: None,
    };
    let resized_image = resized_image.as_view().unwrap();

    let mut image = image::RgbImage::new(IMAGE_SIZE, IMAGE_SIZE);
    let horz_padding = IMAGE_SIZE - resized_image.width();
    let vert_padding = IMAGE_SIZE - resized_image.height();
    image::imageops::overlay(&mut image, &resized_image, (horz_padding / 2) as i64, (vert_padding / 2) as i64);

    ResizingInfo { image, horz_padding, vert_padding }
}

pub(crate) fn predict(session: &ort::Session, original_image: &impl GenericImageWithContinuousBuffer<Rgb<u8>>) -> OrtResult<Vec<Detection>> {
    let s = std::time::Instant::now();
    let ResizingInfo { image, vert_padding, horz_padding } = resize(original_image);
    tracing::info!("resized in {:?}", s.elapsed());

    let s = std::time::Instant::now();
    let input = ndarray::Array4::from_shape_fn(
        (1, 3, IMAGE_SIZE as _, IMAGE_SIZE as _),
        |(_, c, y, x)| {
            image.get_pixel(x as u32, y as u32).0[c] as f32 / 255.0
        }).into_dyn();
    tracing::info!("converted into tensor in {:?}", s.elapsed());

    let input = InputTensor::FloatTensor(input);

    let s = std::time::Instant::now();
    let result = session.run(&[input])?;
    tracing::info!("infer in {:?}", s.elapsed());

    let predictions = result[0].try_extract::<f32>()?;
    let predictions = predictions.view();

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

        let class_id = classes
            .iter()
            .copied()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.total_cmp(&b))
            .map(|it| it.0)
            .unwrap();
        let cx = (cx - horz_padding as f32 / 2.0) * scale;
        let cy = (cy - vert_padding as f32 / 2.0) * scale;
        let w = w * scale;
        let h = h * scale;

        let x = (cx - w / 2.0) as i32;
        let y = (cy - h / 2.0) as i32;

        let rect = Rect::at(x.max(0), y.max(0))
            .of_size(
                (w as u32).min(original_image.width()),
                (h as u32).min(original_image.height()),
            );

        boxes.push(Detection { rect, conf, class_id });
    }

    let filtered = nms(&boxes, 0.1, 3);
    Ok(boxes.into_iter().enumerate().filter_map(|(idx, b)| filtered.contains(&idx).then_some(b)).collect())
}

fn new_size_to_fit_preserving_aspect_ratio(
    image: &impl GenericImageView<Pixel=Rgb<u8>>,
    desired_width: u32,
    desired_height: u32,
) -> (u32, u32) {
    let width = image.width() as f32;
    let height = image.height() as f32;
    let ratio = width / height;
    let (new_width, new_height) = if ratio > 1.0 {
        let scale = width / desired_width as f32;
        (desired_width, (height / scale).round() as u32)
    } else {
        let scale = height / desired_height as f32;
        ((width / scale).round() as u32, desired_height)
    };
    (new_width, new_height)
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
