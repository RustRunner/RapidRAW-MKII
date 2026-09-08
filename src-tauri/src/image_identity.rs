//! Load publication and analysis use the original-image mutex as their shared
//! ownership boundary. Lock order: original_image, then an individual cache.
//! No decoding, analysis, or await belongs inside that boundary.
use crate::app_state::{AppState, LoadedImage};
use crate::noise_analysis::SourceNoiseAnalysis;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageIdentity {
    pub path: String,
    // Opaque decimal string on the wire, avoiding JavaScript integer rounding.
    pub generation: String,
}

impl LoadedImage {
    pub fn identity(&self) -> ImageIdentity {
        ImageIdentity {
            path: self.path.clone(),
            generation: self.generation.to_string(),
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct OwnedEstimate<T> {
    pub identity: ImageIdentity,
    pub estimate: T,
}

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "code", content = "message", rename_all = "snake_case")]
pub enum EstimateError {
    Stale,
    Failed(String),
}

pub type NoiseCache = Option<(ImageIdentity, Arc<SourceNoiseAnalysis>)>;

pub struct ImageSession<'a> {
    ready: &'a Mutex<Option<LoadedImage>>,
    generation: &'a AtomicUsize,
    noise_cache: &'a Mutex<NoiseCache>,
}

impl<'a> ImageSession<'a> {
    pub fn new(state: &'a AppState) -> Self {
        Self {
            ready: &state.original_image,
            generation: &state.load_image_generation,
            noise_cache: &state.noise_estimate_cache,
        }
    }

    pub fn begin_load(&self) -> usize {
        let mut ready = self.ready.lock().unwrap();
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        *ready = None;
        *self.noise_cache.lock().unwrap() = None;
        generation
    }

    pub fn commit_load(&self, image: LoadedImage) -> Result<ImageIdentity, EstimateError> {
        let mut ready = self.ready.lock().unwrap();
        if image.generation != self.generation.load(Ordering::SeqCst) {
            return Err(EstimateError::Stale);
        }
        let identity = image.identity();
        *ready = Some(image);
        Ok(identity)
    }

    pub fn with_generation<T>(
        &self,
        generation: usize,
        work: impl FnOnce() -> T,
    ) -> Result<T, EstimateError> {
        let _ready = self.ready.lock().unwrap();
        if generation != self.generation.load(Ordering::SeqCst) {
            return Err(EstimateError::Stale);
        }
        Ok(work())
    }

    fn with_current<T>(
        &self,
        identity: &ImageIdentity,
        work: impl FnOnce(&LoadedImage) -> T,
    ) -> Result<T, EstimateError> {
        let ready = self.ready.lock().unwrap();
        let image = ready.as_ref().ok_or(EstimateError::Stale)?;
        if image.generation != self.generation.load(Ordering::SeqCst)
            || image.identity() != *identity
        {
            return Err(EstimateError::Stale);
        }
        Ok(work(image))
    }

    pub fn snapshot(&self, expected: &ImageIdentity) -> Result<LoadedImage, EstimateError> {
        self.with_current(expected, Clone::clone)
    }

    pub fn cached_noise(
        &self,
        snapshot: &LoadedImage,
    ) -> Result<Option<Arc<SourceNoiseAnalysis>>, EstimateError> {
        let identity = snapshot.identity();
        self.with_current(&identity, |_| {
            self.noise_cache
                .lock()
                .unwrap()
                .as_ref()
                .filter(|(key, _)| *key == identity)
                .map(|(_, value)| value.clone())
        })
    }

    pub fn publish_noise(
        &self,
        snapshot: &LoadedImage,
        estimate: Arc<SourceNoiseAnalysis>,
    ) -> Result<(), EstimateError> {
        let identity = snapshot.identity();
        self.with_current(&identity, |_| {
            *self.noise_cache.lock().unwrap() = Some((identity.clone(), estimate));
        })
    }

    pub fn finish<T>(
        &self,
        snapshot: &LoadedImage,
        estimate: T,
    ) -> Result<OwnedEstimate<T>, EstimateError> {
        let identity = snapshot.identity();
        self.with_current(&identity, |_| OwnedEstimate {
            identity: identity.clone(),
            estimate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Default)]
    struct State {
        ready: Mutex<Option<LoadedImage>>,
        generation: AtomicUsize,
        cache: Mutex<NoiseCache>,
    }
    impl State {
        fn session(&self) -> ImageSession<'_> {
            ImageSession {
                ready: &self.ready,
                generation: &self.generation,
                noise_cache: &self.cache,
            }
        }
    }
    fn image(path: &str, generation: usize, raw: bool) -> LoadedImage {
        LoadedImage {
            path: path.into(),
            generation,
            is_raw: raw,
            image: Arc::new(image::DynamicImage::new_rgb8(8, 8)),
        }
    }
    fn noise(value: f32) -> Arc<SourceNoiseAnalysis> {
        Arc::new(SourceNoiseAnalysis {
            legacy_source: crate::denoising::NoiseEstimate {
                sigma_luma: value,
                sigma_chroma: value,
                strength: value,
                chroma: value,
            },
            measurement: crate::noise_analysis::NoiseMeasurement {
                version: crate::noise_analysis::VERSION,
                bins: vec![],
                quality: Default::default(),
            },
        })
    }

    #[test]
    fn loads_publish_only_their_committed_generation() {
        let state = State::default();
        let session = state.session();
        let a = session.begin_load();
        let b = session.begin_load();
        let id_b = session.commit_load(image("B", b, true)).unwrap();
        assert_eq!(
            session.commit_load(image("A", a, false)),
            Err(EstimateError::Stale)
        );
        assert_eq!(session.snapshot(&id_b).unwrap().generation, b);
        assert!(session
            .with_generation(a, || panic!("old cache write"))
            .is_err());
    }

    #[test]
    fn cache_hits_and_old_completions_cannot_cross_images_or_reloads() {
        let state = State::default();
        let session = state.session();
        let a = session.begin_load();
        let id_a = session.commit_load(image("A", a, false)).unwrap();
        let snapshot_a = session.snapshot(&id_a).unwrap();
        let shared = noise(1.0);
        session.publish_noise(&snapshot_a, shared.clone()).unwrap();
        assert!(Arc::ptr_eq(
            &session.cached_noise(&snapshot_a).unwrap().unwrap(),
            &shared
        ));
        assert_eq!(
            session
                .cached_noise(&snapshot_a)
                .unwrap()
                .unwrap()
                .legacy_source
                .strength,
            1.0
        );
        let b = session.begin_load();
        assert!(session.snapshot(&id_a).is_err()); // loading is not ready
        let id_b = session.commit_load(image("B", b, true)).unwrap();
        let snapshot_b = session.snapshot(&id_b).unwrap();
        session.publish_noise(&snapshot_b, noise(2.0)).unwrap();
        assert!(session.cached_noise(&snapshot_a).is_err());
        assert_eq!(
            session.publish_noise(&snapshot_a, noise(3.0)),
            Err(EstimateError::Stale)
        );
        assert_eq!(
            session
                .cached_noise(&snapshot_b)
                .unwrap()
                .unwrap()
                .legacy_source
                .strength,
            2.0
        );
        assert!(!snapshot_a.is_raw && snapshot_b.is_raw);
        assert!(!Arc::ptr_eq(&snapshot_a.image, &snapshot_b.image));
        assert!(session.finish(&snapshot_a, "late glare result").is_err());
        let new_a = session.begin_load();
        let new_id_a = session.commit_load(image("A", new_a, false)).unwrap();
        assert_ne!(id_a, new_id_a);
        assert!(session.snapshot(&id_a).is_err());
        let snapshot = session.snapshot(&new_id_a).unwrap();
        assert!(session.cached_noise(&snapshot).unwrap().is_none());
        assert_eq!(session.finish(&snapshot, 42).unwrap().identity, new_id_a);
    }

    #[test]
    fn completion_after_a_real_thread_switch_is_stale() {
        let state = State::default();
        let session = state.session();
        let generation = session.begin_load();
        let id = session.commit_load(image("A", generation, true)).unwrap();
        let snapshot = session.snapshot(&id).unwrap();
        std::thread::scope(|scope| {
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (finish_tx, finish_rx) = std::sync::mpsc::channel();
            let session_ref = &session;
            let worker = scope.spawn(move || {
                started_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
                session_ref.publish_noise(&snapshot, noise(9.0))
            });
            started_rx.recv().unwrap();
            let b = session.begin_load();
            session.commit_load(image("B", b, false)).unwrap();
            finish_tx.send(()).unwrap();
            assert_eq!(worker.join().unwrap(), Err(EstimateError::Stale));
        });
    }

    #[test]
    fn wire_identity_preserves_large_generations_and_stale_is_typed() {
        let id = image("A::virtual-copy", usize::MAX, true).identity();
        let value = serde_json::to_value(&id).unwrap();
        assert_eq!(value["generation"], usize::MAX.to_string());
        assert_eq!(serde_json::from_value::<ImageIdentity>(value).unwrap(), id);
        assert_eq!(
            serde_json::to_value(EstimateError::Stale).unwrap(),
            serde_json::json!({"code":"stale"})
        );
    }
}
