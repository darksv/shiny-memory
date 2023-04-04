#![feature(slice_group_by)]

use std::collections::HashSet;
use std::os::windows::prelude::*;
use std::path::{Path, PathBuf};
use image_hasher::{HasherConfig, ImageHash};
use rayon::prelude::*;

#[derive(Debug)]
struct FileInfo {
    path: PathBuf,
    hash: ImageHash,
}

fn hash_images(path: impl AsRef<Path>) -> Vec<FileInfo> {
    let mut items: Vec<_> = std::fs::read_dir(path).unwrap()
        .par_bridge()
        .flat_map(|item| {
            let item = item.unwrap();
            let metadata = item.metadata().unwrap();
            if !metadata.is_file() {
                return None;
            }

            let image = match image::open(item.path()) {
                Ok(img) => img,
                Err(e) => {
                    println!("{}: {:?}", item.path().display(), e);
                    return None;
                }
            };

            let hasher = HasherConfig::new()
                .hash_size(8, 8)
                .to_hasher();
            let hash = hasher.hash_image(&image);

            Some(FileInfo { path: item.path(), hash })
        })
        .collect();

    items
}

fn main() {
    let used: HashSet<ImageHash> = hash_images(r"D:\label-studio\media\upload\1").into_iter().map(|it| it.hash).collect();
    let items = hash_images(r"D:\yolo\tts3\.uniq4");

    let mut moved = HashSet::new();

    let mut uniq = 0;
    for item in &items {
        if !used.contains(&item.hash) && moved.insert(&item.hash) {
            uniq += 1;

            let new = item.path.parent().unwrap().join(".uniq").join(item.path.file_name().unwrap());
            std::fs::create_dir_all(new.parent().unwrap()).unwrap();
            std::fs::rename(&item.path, new).unwrap();
        }
    }

    println!("{} / {}", uniq, items.len());
}

fn dedup(items: &mut [FileInfo]) {
    items.sort_by_key(|fi| fi.hash.to_base64());
    for group in items.group_by(|a, b| a.hash.dist(&b.hash) == 0) {
        if group.len() > 1 {
            find_best(group);
        }
    }
}

fn find_best(fi: &[FileInfo]) {
    let best = &fi[0];
    for f in &fi[1..] {
        let fs1 = std::fs::metadata(&best.path).unwrap().file_size();
        let fs2 = std::fs::metadata(&f.path).unwrap().file_size();
        if fs1 == fs2 {
            continue;
        }

        let i1 = image::open(&best.path).unwrap();
        let i2 = image::open(&f.path).unwrap();
        if i1.width() * i1.height() >= i2.width() * i2.height() {
            continue;
        }
    }

    for f in fi {
        if f.path != best.path {
            let new = f.path.parent().unwrap().join(".dups").join(f.path.file_name().unwrap());
            std::fs::create_dir_all(new.parent().unwrap()).unwrap();
            std::fs::rename(&f.path, new).unwrap();
        }
    }
}
