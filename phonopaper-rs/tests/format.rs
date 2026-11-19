use phonopaper_rs::format::{
    HIGH_FREQ, LOW_FREQ, MULTITONES, TOTAL_BINS, freq_to_index, index_to_freq,
};

#[test]
fn test_index_zero_is_high_freq() {
    let f = index_to_freq(0);
    assert!(
        (f - HIGH_FREQ).abs() < 1.0,
        "index 0 should be ~{HIGH_FREQ} Hz, got {f}"
    );
}

#[test]
fn test_last_index_is_low_freq() {
    let f = index_to_freq(TOTAL_BINS - 1);
    // bin 383: diff = 251 - 383 = -132, freq = 2^(-132/48) * 440 = 65.406 Hz (C2)
    assert!(
        (f - LOW_FREQ).abs() < 0.01,
        "last index should be ~{LOW_FREQ} Hz, got {f}"
    );
}

#[test]
fn test_round_trip() {
    for i in [0, 10, 100, 200, 300, TOTAL_BINS - 1] {
        let f = index_to_freq(i);
        let j = freq_to_index(f);
        assert_eq!(
            i, j,
            "round-trip failed for index {i}: freq={f}, recovered={j}"
        );
    }
}

#[test]
fn test_a4_440hz() {
    // A4 = 440 Hz is 33 semitones (+ 2 octaves of grid) above C2 at bin 383:
    // 383 − 33 × 4 = 251.
    let expected_index = TOTAL_BINS - 1 - 33 * MULTITONES;
    let idx = freq_to_index(440.0);
    assert_eq!(
        idx, expected_index,
        "440 Hz should map to index {expected_index}"
    );
}

/// Regression: the bottom row of the grid is C2 = 65.406 Hz, the value of
/// `PHONOPAPER_LOWEST_FREQ` in the official `PhonoPaper` / Virtual ANS
/// programs and "from 65.4Hz" in the official specification image.  The
/// previous pivot (252) put every bin ≈ 25 cents sharp.
#[test]
fn grid_matches_official_lowest_frequency() {
    // f32 constant stored in the official bytecode.
    let official = f64::from(65.406_36_f32);
    assert!((LOW_FREQ - official).abs() < 1e-4, "LOW_FREQ = {LOW_FREQ}");
    assert!((index_to_freq(TOTAL_BINS - 1) - official).abs() < 1e-4);
    // Every 48 bins up is exactly one octave.
    for octave in 0..8 {
        let f = index_to_freq(TOTAL_BINS - 1 - 48 * octave);
        let expected = official * f64::from(1u32 << octave);
        assert!(
            (f / expected - 1.0).abs() < 1e-6,
            "C{} should be {expected} Hz, got {f}",
            octave + 2
        );
    }
    // Equal-tempered notes land exactly on a bin (multiple of 4).
    for (freq, name) in [(440.0, "A4"), (261.625_565, "C4"), (4186.009, "C8")] {
        let idx = freq_to_index(freq);
        assert_eq!(
            (TOTAL_BINS - 1 - idx) % MULTITONES,
            0,
            "{name} at bin {idx}"
        );
        let cents = 1200.0 * (index_to_freq(idx) / freq).log2();
        assert!(cents.abs() < 0.1, "{name} is {cents} cents off its bin");
    }
}
