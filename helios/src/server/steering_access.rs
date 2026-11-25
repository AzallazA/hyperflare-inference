/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional per-request access gate for activation steering (`steering_scale`).
//!
//! Without configured steering keys every client may steer (the pre-gate behaviour). With keys,
//! a request whose body sets a nonzero `steering_scale` (top level or in `chat_template_kwargs`)
//! must present one of them in `X-Steering-Key: <key>`, or, without that header, in
//! `Authorization: Bearer <key>`; otherwise the server answers 403 before preparing the request.
//! `steering_scale` 0 or omitted never needs a key. Keys are kept only as SHA-256 digests and are
//! never written to logs or error messages.

use axum::http::HeaderMap;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Header that carries a steering key independently of the serving API key's `Authorization`.
pub(super) const STEERING_KEY_HEADER: &str = "x-steering-key";

/// The message of the 403 answer; it names the headers, never a key.
pub(super) const FORBIDDEN_MESSAGE: &str = "steering_scale requires a steering key on this server: \
    send it as X-Steering-Key: <key> or Authorization: Bearer <key>, or omit steering_scale (or set it to 0)";

/// Configured steering keys, held as SHA-256 digests.
#[derive(Clone, Default)]
pub(super) struct SteeringAccess {
    digests: Vec<[u8; 32]>,
}

impl std::fmt::Debug for SteeringAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SteeringAccess({} keys)", self.digests.len())
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Compares two digests without an early exit.
fn digests_equal(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter().zip(right).fold(0u8, |difference, (a, b)| difference | (a ^ b)) == 0
}

impl SteeringAccess {
    pub(super) fn new<S: AsRef<str>>(keys: &[S]) -> Self {
        Self { digests: keys.iter().map(|key| digest(key.as_ref().as_bytes())).collect() }
    }

    /// True when the server restricts steering to key holders.
    pub(super) fn restricted(&self) -> bool {
        !self.digests.is_empty()
    }

    /// True when `presented` matches a configured key; every key is compared, and the
    /// comparison runs over fixed-length digests, so timing reveals neither which key matched
    /// nor the length of any key.
    fn matches(&self, presented: &[u8]) -> bool {
        let presented = digest(presented);
        self.digests
            .iter()
            .fold(false, |found, key| found | digests_equal(key, &presented))
    }

    /// Whether the request's headers authorize steering. Unrestricted servers authorize every
    /// request. `X-Steering-Key` takes precedence when present (so a serving key in
    /// `Authorization` and a different steering key can travel together); otherwise a
    /// `Bearer` credential in `Authorization` is checked against the steering keys.
    pub(super) fn authorized(&self, headers: &HeaderMap) -> bool {
        if !self.restricted() {
            return true;
        }
        if let Some(value) = headers.get(STEERING_KEY_HEADER) {
            return self.matches(value.as_bytes());
        }
        headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.as_bytes().strip_prefix(b"Bearer "))
            .is_some_and(|key| self.matches(key))
    }
}

fn nonzero(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        // A number that is exactly zero asks for no steering; anything else (including a
        // malformed value, which decoding would reject) asks for steering and needs a key.
        Some(Value::Number(number)) => number.as_f64() != Some(0.),
        Some(_) => true,
    }
}

/// Whether a decoded request body asks for steering: a non-null `steering_scale` other than 0,
/// at the top level or in `chat_template_kwargs`.
pub(super) fn requested(body: &Value) -> bool {
    nonzero(body.get("steering_scale"))
        || nonzero(
            body.get("chat_template_kwargs")
                .and_then(|kwargs| kwargs.get("steering_scale")),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().unwrap());
        }
        map
    }

    #[test]
    fn steering_requests_are_detected_at_top_level_and_in_template_kwargs() {
        for (body, expected) in [
            (json!({}), false),
            (json!({"steering_scale": null}), false),
            (json!({"steering_scale": 0}), false),
            (json!({"steering_scale": 0.0}), false),
            (json!({"steering_scale": -0.0}), false),
            (json!({"steering_scale": 1}), true),
            (json!({"steering_scale": 0.25}), true),
            (json!({"steering_scale": -1}), true),
            (json!({"steering_scale": "1"}), true),
            (json!({"chat_template_kwargs": {"steering_scale": 0}}), false),
            (json!({"chat_template_kwargs": {"steering_scale": 0.5}}), true),
            (json!({"chat_template_kwargs": {"enable_thinking": true}}), false),
            (json!({"chat_template_kwargs": "x"}), false),
        ] {
            assert_eq!(requested(&body), expected, "{body}");
        }
    }

    #[test]
    fn unrestricted_access_authorizes_everything() {
        let access = SteeringAccess::new::<&str>(&[]);
        assert!(!access.restricted());
        assert!(access.authorized(&headers(&[])));
        assert!(access.authorized(&headers(&[("x-steering-key", "anything")])));
    }

    #[test]
    fn keys_are_checked_from_either_header_with_the_dedicated_header_first() {
        let access = SteeringAccess::new(&["key-one", "key-two"]);
        assert!(access.restricted());
        assert!(!access.authorized(&headers(&[])));
        assert!(access.authorized(&headers(&[("x-steering-key", "key-one")])));
        assert!(access.authorized(&headers(&[("x-steering-key", "key-two")])));
        assert!(!access.authorized(&headers(&[("x-steering-key", "key-three")])));
        assert!(!access.authorized(&headers(&[("x-steering-key", "key-one ")])));
        assert!(!access.authorized(&headers(&[("x-steering-key", "")])));
        assert!(access.authorized(&headers(&[("authorization", "Bearer key-two")])));
        assert!(!access.authorized(&headers(&[("authorization", "Bearer key-thre")])));
        assert!(!access.authorized(&headers(&[("authorization", "key-one")])));
        assert!(!access.authorized(&headers(&[("authorization", "Basic key-one")])));
        // A serving key in Authorization and the steering key in X-Steering-Key travel together.
        assert!(access.authorized(&headers(&[
            ("authorization", "Bearer serving-key"),
            ("x-steering-key", "key-one"),
        ])));
        // A wrong dedicated header is not rescued by a matching Bearer credential.
        assert!(!access.authorized(&headers(&[
            ("authorization", "Bearer key-one"),
            ("x-steering-key", "wrong"),
        ])));
    }

    #[test]
    fn debug_and_forbidden_message_carry_no_key() {
        let access = SteeringAccess::new(&["secret-steering-key"]);
        assert!(!format!("{access:?}").contains("secret"));
        assert!(!FORBIDDEN_MESSAGE.contains("secret"));
    }
}
