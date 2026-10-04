//! The gateway console: one embedded HTML document (`app.html`) served at
//! `/web/`.
//!
//! It replaced two pages (`dashboard.html` + `settings.html`) that had duplicated
//! the header, the styles and the fetch layer, and had already drifted apart.
//! Analysis of what was wrong and why each piece looks the way it does now is
//! in `docs/gui-rework-analysis.md`.
