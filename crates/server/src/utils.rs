use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};

pub(crate) struct TempFilePath {
    path: PathBuf,
}

impl TempFilePath {
    pub(crate) fn new(dir: impl AsRef<Path>) -> Result<Self, io::Error> {
        let unique = RandomState::new().build_hasher().finish();
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{unique}.bin"));
        tracing::info!("creating a temporary file {}", path.display());
        Ok(Self { path })
    }
}

impl AsRef<Path> for TempFilePath {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFilePath {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.path) {
            Ok(..) => tracing::info!("deleted {}", self.path.display()),
            Err(e) => tracing::info!("deleting {} error: {:?}", self.path.display(), e),
        }
    }
}

impl Deref for TempFilePath {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}