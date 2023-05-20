use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};

pub(crate) struct TempFile {
    path: PathBuf,
}

impl TempFile {
    pub(crate) fn new(dir: impl AsRef<Path>) -> Result<Self, io::Error> {
        let unique = RandomState::new().build_hasher().finish();
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{unique}.bin"));
        Ok(Self { path })
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.path) {
            Ok(..) => tracing::info!("deleted {}", self.path.display()),
            Err(e) => tracing::info!("deleting {} error: {:?}", self.path.display(), e),
        }
    }
}

impl Deref for TempFile {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}