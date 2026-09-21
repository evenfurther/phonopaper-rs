//! Ground-truth file formats: `labels.csv` and `manifest.json`.
//!
//! `labels.csv` has one header line followed by one row per image:
//!
//! ```text
//! file,present,x0,y0,x1,y1,x2,y2,x3,y3
//! 000000.png,1,12.35,40.10,110.02,38.77,108.90,95.12,13.40,97.00
//! 000001.png,0,0,0,0,0,0,0,0,0
//! ```
//!
//! * `present` is `1` when the image contains a `PhonoPaper` pattern.
//! * `(x0,y0) … (x3,y3)` are the pattern's top-left, top-right, bottom-right
//!   and bottom-left corners **in pattern orientation**, in pixels of the
//!   image (origin at the top-left corner, pixel centres at `+0.5`).
//!   Corners may lie slightly outside the image.  All eight values are `0`
//!   for negative samples.
//!
//! Coordinates are written with two decimals so the file is byte-stable.

use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::geometry::{Point, Quad};
use crate::sample::GeneratorConfig;

/// Header line of `labels.csv`.
pub const CSV_HEADER: &str = "file,present,x0,y0,x1,y1,x2,y2,x3,y3";

/// One row of `labels.csv`.
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    /// Image file name, relative to the dataset directory.
    pub file: String,
    /// Corners of the pattern, or `None` for a negative sample.
    pub corners: Option<Quad>,
}

impl Label {
    /// Format as one CSV row (without trailing newline).
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        let mut row = String::new();
        match &self.corners {
            Some(q) => {
                let _ = write!(row, "{},1", self.file);
                for p in q {
                    let _ = write!(row, ",{:.2},{:.2}", p.x, p.y);
                }
            }
            None => {
                let _ = write!(row, "{},0,0,0,0,0,0,0,0,0", self.file);
            }
        }
        row
    }

    /// Parse one CSV row.
    ///
    /// # Errors
    ///
    /// Returns a message when the row does not have ten fields or a field
    /// cannot be parsed.
    pub fn from_csv_row(row: &str) -> Result<Self, String> {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() != 10 {
            return Err(format!("expected 10 fields, got {}: {row:?}", fields.len()));
        }
        let file = fields[0].to_owned();
        let present = match fields[1] {
            "0" => false,
            "1" => true,
            other => return Err(format!("invalid presence flag {other:?}")),
        };
        if !present {
            return Ok(Self {
                file,
                corners: None,
            });
        }
        let mut vals = [0.0_f64; 8];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = fields[2 + i]
                .parse()
                .map_err(|e| format!("bad coordinate {:?}: {e}", fields[2 + i]))?;
        }
        Ok(Self {
            file,
            corners: Some([
                Point::new(vals[0], vals[1]),
                Point::new(vals[2], vals[3]),
                Point::new(vals[4], vals[5]),
                Point::new(vals[6], vals[7]),
            ]),
        })
    }
}

/// Serialise labels to the full CSV text (header + rows + trailing newline).
#[must_use]
pub fn labels_to_csv(labels: &[Label]) -> String {
    let mut out = String::with_capacity(labels.len() * 80);
    out.push_str(CSV_HEADER);
    out.push('\n');
    for label in labels {
        out.push_str(&label.to_csv_row());
        out.push('\n');
    }
    out
}

/// Parse a full CSV text.
///
/// # Errors
///
/// Returns a message on a missing/invalid header or an invalid row.
pub fn labels_from_csv(text: &str) -> Result<Vec<Label>, String> {
    let mut lines = text.lines();
    match lines.next() {
        Some(h) if h.trim() == CSV_HEADER => {}
        other => return Err(format!("invalid header {other:?}")),
    }
    lines
        .filter(|l| !l.trim().is_empty())
        .map(Label::from_csv_row)
        .collect()
}

/// Read `labels.csv` from a dataset directory.
///
/// # Errors
///
/// Returns a message on I/O or parse failure.
pub fn read_labels(dir: &Path) -> Result<Vec<Label>, String> {
    let text = std::fs::read_to_string(dir.join("labels.csv")).map_err(|e| e.to_string())?;
    labels_from_csv(&text)
}

/// Metadata written next to the images as `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    /// Format version; bump when the CSV layout or semantics change.
    pub format_version: u32,
    /// Generator parameters used.
    pub config: GeneratorConfig,
    /// Human-readable description of the corner convention.
    pub corner_order: String,
    /// Number of positive samples.
    pub positives: u64,
    /// Number of negative samples.
    pub negatives: u64,
}

/// Read `manifest.json` from a dataset directory.
///
/// # Errors
///
/// Returns a message on I/O or parse failure.
pub fn read_manifest(dir: &Path) -> Result<Manifest, String> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::rect_quad;

    #[test]
    fn csv_round_trip() {
        let labels = vec![
            Label {
                file: "000000.png".into(),
                corners: Some(rect_quad(1.25, 2.5, 30.0, 40.75)),
            },
            Label {
                file: "000001.png".into(),
                corners: None,
            },
        ];
        let text = labels_to_csv(&labels);
        assert!(text.starts_with(CSV_HEADER));
        assert_eq!(
            text,
            "file,present,x0,y0,x1,y1,x2,y2,x3,y3\n000000.png,1,1.25,2.50,30.00,2.50,30.00,40.75,1.25,40.75\n000001.png,0,0,0,0,0,0,0,0,0\n"
        );
        assert_eq!(labels_from_csv(&text).unwrap(), labels);
    }

    #[test]
    fn invalid_rows_are_rejected() {
        assert!(Label::from_csv_row("a,1,2").is_err());
        assert!(Label::from_csv_row("a,2,0,0,0,0,0,0,0,0").is_err());
        assert!(Label::from_csv_row("a,1,x,0,0,0,0,0,0,0").is_err());
        assert!(labels_from_csv("wrong header\n").is_err());
    }
}
