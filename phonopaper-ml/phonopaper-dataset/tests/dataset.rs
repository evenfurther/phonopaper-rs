//! End-to-end determinism of the dataset generator.

use phonopaper_dataset::labels::{read_labels, read_manifest};
use phonopaper_dataset::{GeneratorConfig, generate_dataset, image_file_name};

fn small_config() -> GeneratorConfig {
    GeneratorConfig {
        count: 12,
        size: 48,
        seed: 99,
        positive_ratio: 0.6,
    }
}

#[test]
fn two_runs_produce_identical_files() {
    let cfg = small_config();
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let ma = generate_dataset(&cfg, a.path()).unwrap();
    let mb = generate_dataset(&cfg, b.path()).unwrap();
    assert_eq!(ma, mb);
    assert_eq!(ma.positives + ma.negatives, cfg.count);

    for name in ["labels.csv", "manifest.json"] {
        assert_eq!(
            std::fs::read(a.path().join(name)).unwrap(),
            std::fs::read(b.path().join(name)).unwrap(),
            "{name} differs"
        );
    }
    for idx in 0..cfg.count {
        let name = image_file_name(idx);
        assert_eq!(
            std::fs::read(a.path().join(&name)).unwrap(),
            std::fs::read(b.path().join(&name)).unwrap(),
            "{name} differs"
        );
    }
}

#[test]
fn labels_and_manifest_are_readable_and_consistent() {
    let cfg = small_config();
    let dir = tempfile::tempdir().unwrap();
    let manifest = generate_dataset(&cfg, dir.path()).unwrap();
    let labels = read_labels(dir.path()).unwrap();
    assert_eq!(labels.len(), usize::try_from(cfg.count).unwrap());
    assert_eq!(read_manifest(dir.path()).unwrap(), manifest);
    let positives = labels.iter().filter(|l| l.corners.is_some()).count() as u64;
    assert_eq!(positives, manifest.positives);
    for (idx, label) in labels.iter().enumerate() {
        assert_eq!(label.file, image_file_name(u64::try_from(idx).unwrap()));
        let img = image::open(dir.path().join(&label.file)).unwrap();
        assert_eq!(img.width(), cfg.size);
        assert_eq!(img.height(), cfg.size);
    }
}
