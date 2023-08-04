use imageproc::drawing::Canvas;
use imageproc::rect::Rect;

pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn mid(self, other: Self) -> Self {
        Self {
            x: 0.5 * (self.x + other.x),
            y: 0.5 * (self.y + other.y),
        }
    }
}

pub struct Size {
    pub width: f32,
    pub height: f32,
}

pub struct RotatedRect {
    pub center: Point,
    pub size: Size,
    pub angle: f32,
}

impl RotatedRect {
    pub fn bounds(&self) -> Rect {
        Rect::at(
            (self.center.x - self.size.width / 2.0).round() as _,
            (self.center.y - self.size.height / 2.0).round() as _
        ).of_size(self.size.width.round() as _, self.size.height.round() as _)
    }
}

pub fn nms_boxes(boxes: &[impl DetectedBox], overlap_threshold: f32, neighbour_threshold: usize) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut indices = Vec::with_capacity(boxes.len());
    indices.extend(0..boxes.len());
    indices.sort_by_key(|&idx| (boxes[idx].rect().bottom(), idx));

    let mut remaining_boxes = Vec::new();

    while let Some(idx1) = indices.pop() {
        let box1 = &boxes[idx1];

        let mut neighbours = 0;
        indices.retain(|idx2| {
            let box2 = &boxes[*idx2];
            if box1.class() != box2.class() {
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

pub trait DetectedBox {
    fn rect(&self) -> Rect;

    fn conf(&self) -> f32;

    fn class(&self) -> Option<usize> {
        None
    }

    fn area(&self) -> u32 {
        (self.rect().width() + 1) * (self.rect().height() + 1)
    }

    fn intersection_area(&self, other: &Self) -> u32 {
        self.rect().intersect(other.rect()).map_or(0, |r| {
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