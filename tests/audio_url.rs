//! The audio URL composition convention, and the 404 it used to produce.
//!
//! History, because a test that only describes the present state is a test
//! that cannot explain itself:
//!
//! * `72d287a` (2026-07-20) introduced `STT_URL` / `TTS_URL` with the default
//!   `http://netai-stt:5092` — **no** `/v1/audio` suffix. The byte proxy
//!   stripped the prefix from the request path, so base+leaf composed
//!   correctly.
//! * `95e340b` (2026-07-20) added a second audio path that *appended* the
//!   full `/v1/audio/transcriptions`. Consistent with the suffix-free default,
//!   so it composed correctly too.
//! * `2c6ed0b` (2026-08-25) added the `/v1/audio` suffix to fix the byte
//!   proxy — its message says so verbatim: *"so the handler's prefix-strip
//!   composes to real upstream URLs (was /transcriptions -> 404)"*. That
//!   silently turned the appending path into
//!   `/v1/audio/v1/audio/transcriptions` → 404.
//! * `b4dabfb` (2026-08-29, NetAI-Stack) put the suffix into
//!   `STT_URL=http://speech-stt:5092/v1/audio` in `compose/base.yml` for the
//!   same reason, so the default stopped being the one that shipped.
//!
//! Same variable, two conventions — strip vs. append. A fix for one broke the
//! other, and nothing noticed for 36 days because **nothing tested the
//! composition**. `audio_target()` makes all four combinations converge, so
//! the shape of the bug is gone rather than merely covered.

#[cfg(test)]
mod audio_target_tests {
    use cascade_llm::audio::{audio_target, AUDIO_PREFIX};

    /// The route `netai-stt` actually serves
    /// (`config/parakeet-tdt-0.6b-v3-fastapi-openai/app.py`).
    const REAL_STT_ROUTE: &str = "http://speech-stt:5092/v1/audio/transcriptions";
    const REAL_TTS_ROUTE: &str = "http://speech-stt:5092/v1/audio/speech";

    /// The value `compose/base.yml` ships today.
    const CONFIGURED_WITH_SUFFIX: &str = "http://speech-stt:5092/v1/audio";
    /// The value the binary defaulted to between 2026-07-20 and 2026-08-25.
    const DEFAULT_WITHOUT_SUFFIX: &str = "http://speech-stt:5092";

    #[test]
    fn configured_base_yields_the_real_route() {
        assert_eq!(
            audio_target(CONFIGURED_WITH_SUFFIX, "/v1/audio/transcriptions"),
            REAL_STT_ROUTE
        );
    }

    #[test]
    fn suffix_free_base_yields_the_same_route() {
        assert_eq!(
            audio_target(DEFAULT_WITHOUT_SUFFIX, "/v1/audio/transcriptions"),
            REAL_STT_ROUTE
        );
    }

    #[test]
    fn leaf_only_request_path_yields_the_same_route() {
        assert_eq!(
            audio_target(CONFIGURED_WITH_SUFFIX, "/transcriptions"),
            REAL_STT_ROUTE
        );
        assert_eq!(
            audio_target(DEFAULT_WITHOUT_SUFFIX, "/transcriptions"),
            REAL_STT_ROUTE
        );
    }

    /// ★ The regression itself. Before the refactor this was
    /// `http://speech-stt:5092/v1/audio/v1/audio/transcriptions` → 404, and
    /// it returned **HTTP 500 in 14 ms** to the caller, labelled
    /// "Audio transcription failed" — the real cause (a 404 on the upstream)
    /// appeared only in the server log.
    #[test]
    fn a_double_prefix_is_unrepresentable() {
        let bases = [CONFIGURED_WITH_SUFFIX, DEFAULT_WITHOUT_SUFFIX];
        let paths = [
            "/v1/audio/transcriptions",
            "/transcriptions",
            "/v1/audio/transcriptions/",
            "v1/audio/transcriptions",
        ];
        for base in bases {
            for path in paths {
                assert_eq!(
                    audio_target(base, path),
                    REAL_STT_ROUTE,
                    "base={base:?} path={path:?} produced a different URL"
                );
            }
        }
    }

    #[test]
    fn tts_composes_the_same_way() {
        assert_eq!(
            audio_target(CONFIGURED_WITH_SUFFIX, "/v1/audio/speech"),
            REAL_TTS_ROUTE
        );
        assert_eq!(
            audio_target(DEFAULT_WITHOUT_SUFFIX, "/v1/audio/speech"),
            REAL_TTS_ROUTE
        );
    }

    #[test]
    fn trailing_slashes_do_not_change_the_result() {
        assert_eq!(
            audio_target("http://speech-stt:5092/v1/audio/", "/v1/audio/transcriptions"),
            REAL_STT_ROUTE
        );
        assert_eq!(
            audio_target("http://speech-stt:5092//", "//v1/audio/transcriptions"),
            REAL_STT_ROUTE
        );
    }

    /// The prefix is a constant, not a literal sprinkled at each call site —
    /// which is the whole point of having it here.
    #[test]
    fn every_result_carries_exactly_one_audio_prefix() {
        for base in [CONFIGURED_WITH_SUFFIX, DEFAULT_WITHOUT_SUFFIX] {
            let got = audio_target(base, "/v1/audio/transcriptions");
            assert_eq!(got.matches(AUDIO_PREFIX).count(), 1, "got {got}");
            assert!(!got.contains("/v1/audio/v1/audio"), "got {got}");
        }
    }

    /// A port-qualified host must survive normalisation. `netai-stt:5092`
    /// contains a colon, and a naive "strip everything after the last slash"
    /// would take the host with it.
    #[test]
    fn the_host_and_port_are_preserved() {
        let got = audio_target(CONFIGURED_WITH_SUFFIX, "/v1/audio/transcriptions");
        assert!(got.starts_with("http://speech-stt:5092/"), "got {got}");
    }

    /// No other module may compose an audio URL by string concatenation.
    /// `2c6ed0b` and `95e340b` disagreed on one variable, and both patterns
    /// were individually plausible; this asserts the *absence* of the second
    /// one, which a behaviour test cannot do.
    ///
    /// `src/audio.rs` is deliberately **excluded**: it is the definition site,
    /// and its doc comment quotes both routes as the worked example. Grepping
    /// the file that documents the convention against the convention would
    /// fail for the right reason — a guard that misfires teaches you to
    /// ignore it.
    #[test]
    fn no_other_module_rebuilds_an_audio_url_by_hand() {
        let sources = [
            ("handlers.rs", include_str!("../src/handlers.rs")),
            ("media.rs", include_str!("../src/media.rs")),
        ];
        for (name, src) in sources {
            assert!(
                !src.contains("/v1/audio/transcriptions"),
                "{name} builds an STT URL by hand — use audio_target()"
            );
            assert!(
                !src.contains("/v1/audio/speech"),
                "{name} builds a TTS URL by hand — use audio_target()"
            );
        }
    }

    /// The deleted code is named here on purpose: `transcribe_audio` in
    /// handlers.rs appended `{stt_url}/v1/audio/transcriptions`. If it ever
    /// comes back — as a feature someone wants — it must go through
    /// `audio_target()` instead, or it is broken again.
    #[test]
    fn the_removed_helper_stays_removed() {
        let handlers = include_str!("../src/handlers.rs");
        assert!(
            !handlers.contains("transcribe_audio"),
            "transcribe_audio was the append-side twin; it must not return as \
             a second audio entry point — use audio_target() from a handler that \
             delegates to proxy_request()"
        );
    }
}