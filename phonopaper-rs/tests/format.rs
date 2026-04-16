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
    // bin 383: diff = 63*4 - 383 = -131, freq = 2^(-131/48) * 440 ≈ 66.36 Hz
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
    // A4 = 440 Hz should map to index 63 * 4 = 252 (the pivot bin)
    let expected_index = 63 * MULTITONES;
    let idx = freq_to_index(440.0);
    assert_eq!(
        idx, expected_index,
        "440 Hz should map to index {expected_index}"
    );
}
