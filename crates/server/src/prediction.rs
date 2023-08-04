use std::num::NonZeroU32;

use fast_image_resize as fr;
use image::{GenericImageView, Pixel, Rgb};
use image::flat::SampleLayout;
use imageproc::rect::Rect;
use ndarray::Axis;
use ort::OrtResult;
use ort::tensor::InputTensor;

use math::DetectedBox;

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

fn image_into_tensor<const N: u32>(image: &image::RgbImage) -> Option<ndarray::Array4<f32>> {
    if image.dimensions() != (N, N) {
        return None;
    }

    let mut out: Vec<f32> = vec![0.0; (N * N * 3) as usize];
    for (x, y, &Rgb([r, g, b])) in image.enumerate_pixels() {
        out[(0 * N * N + N * y + x) as usize] = r as f32 / 255.0;
        out[(1 * N * N + N * y + x) as usize] = g as f32 / 255.0;
        out[(2 * N * N + N * y + x) as usize] = b as f32 / 255.0;
    }

    ndarray::Array4::from_shape_vec((1, 3, N as usize, N as usize), out).ok()
}

pub(crate) fn predict(session: &ort::Session, original_image: &impl GenericImageWithContinuousBuffer<Rgb<u8>>, frame_no: usize) -> OrtResult<Vec<Detection>> {
    let s = std::time::Instant::now();
    let ResizingInfo { image, vert_padding, horz_padding } = resize(original_image);
    let resizing_time = s.elapsed();

    let s = std::time::Instant::now();
    let input = image_into_tensor::<IMAGE_SIZE>(&image)
        .expect("invalid resizing?")
        .into_dyn();
    let conversion_time = s.elapsed();

    let input = InputTensor::FloatTensor(input);

    let s = std::time::Instant::now();
    let result = session.run(&[input])?;
    let inference_time = s.elapsed();

    tracing::info!("#{frame_no:>04} | resize: {resizing_time:>10.3?} | convert: {conversion_time:>10.3?} | infer: {inference_time:>10.3?}");

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

    let filtered = math::nms_boxes(&boxes, 0.1, 3);
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

#[derive(Debug)]
pub(crate) struct Detection {
    pub(crate) rect: Rect,
    pub(crate) conf: f32,
    pub(crate) class_id: usize,
}

impl DetectedBox for Detection {
    fn rect(&self) -> Rect {
        self.rect
    }

    fn conf(&self) -> f32 {
        self.conf
    }

    fn class(&self) -> Option<usize> {
        Some(self.class_id)
    }
}