/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Resolves the steering keys of `flare run` (see `cli::SteeringAccessOptions`). Errors and
//! log lines name sources and counts only, never a key.

use std::path::Path;

use crate::cli::SteeringAccessOptions;

/// Comma-separated steering keys.
pub const KEYS_ENV: &str = "FLARE_STEERING_API_KEYS";
/// File of steering keys; preferred over the other sources because it keeps keys out of the
/// process list and the environment.
pub const KEYS_FILE_ENV: &str = "FLARE_STEERING_API_KEYS_FILE";

/// Splits `text` into keys at commas and line ends; blank entries and lines starting with '#'
/// are skipped. A key must be printable ASCII without spaces.
fn parse(text: &str, source: &str) -> Result<Vec<String>, String> {
    let mut keys = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        for key in line.split(',').map(str::trim).filter(|key| !key.is_empty()) {
            if !key.bytes().all(|byte| byte.is_ascii_graphic()) {
                return Err(format!(
                    "{source}: steering keys must be printable ASCII without spaces (key {} of the source)",
                    keys.len() + 1
                ));
            }
            keys.push(key.to_owned());
        }
    }
    Ok(keys)
}

fn read_file(path: &Path, source: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("{source} {}: cannot read steering keys: {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            if metadata.permissions().mode() & 0o077 != 0 {
                eprintln!(
                    "steering: warning: {} is readable by other users; chmod 600 it",
                    path.display()
                );
            }
        }
    }
    let keys = parse(&text, &format!("{source} {}", path.display()))?;
    if keys.is_empty() {
        // A named key file that holds nothing must not silently leave steering open.
        return Err(format!("{source} {}: no steering key in the file", path.display()));
    }
    Ok(keys)
}

/// Gathers the keys of every configured source: `--steering-api-key`, `--steering-api-keys-file`,
/// `FLARE_STEERING_API_KEYS` and `FLARE_STEERING_API_KEYS_FILE` (`environment` reads a variable).
/// Duplicates are dropped; an empty result means steering stays open to every client.
pub fn resolve(
    options: &SteeringAccessOptions,
    environment: impl Fn(&str) -> Option<String>,
) -> Result<Vec<String>, String> {
    let mut keys = Vec::new();
    for (index, key) in options.steering_api_key.iter().enumerate() {
        keys.extend(parse(&key.0, &format!("--steering-api-key #{}", index + 1))?);
    }
    if options.steering_api_key.iter().any(|key| key.0.trim().is_empty()) {
        return Err("--steering-api-key must not be empty".into());
    }
    if let Some(path) = &options.steering_api_keys_file {
        keys.extend(read_file(path, "--steering-api-keys-file")?);
    }
    if let Some(value) = environment(KEYS_ENV).filter(|value| !value.trim().is_empty()) {
        keys.extend(parse(&value, KEYS_ENV)?);
    }
    if let Some(path) = environment(KEYS_FILE_ENV).filter(|value| !value.trim().is_empty()) {
        keys.extend(read_file(Path::new(path.trim()), KEYS_FILE_ENV)?);
    }
    let mut unique = Vec::with_capacity(keys.len());
    for key in keys {
        if !unique.contains(&key) {
            unique.push(key);
        }
    }
    Ok(unique)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::SteeringKey;
    use std::collections::HashMap;

    const SECRET: &str = "s3cret-steering-key";

    fn env(pairs: &[(&str, String)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> =
            pairs.iter().map(|(name, value)| ((*name).to_owned(), value.clone())).collect();
        move |name| map.get(name).cloned()
    }

    fn file(name: &str, contents: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("flare-steering-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        path
    }

    #[test]
    fn no_source_leaves_steering_open() {
        assert!(resolve(&SteeringAccessOptions::default(), env(&[])).unwrap().is_empty());
        let blank = [(KEYS_ENV, "  ".to_string()), (KEYS_FILE_ENV, String::new())];
        assert!(resolve(&SteeringAccessOptions::default(), env(&blank)).unwrap().is_empty());
    }

    #[test]
    fn every_source_contributes_and_duplicates_collapse() {
        let path = file("keys", &format!("# steering keys\n{SECRET}\n\nfile-two, file-three\n"));
        let env_path = file("env-keys", "env-file-key\n");
        let options = SteeringAccessOptions {
            steering_api_key: vec![SteeringKey("flag-key".into()), SteeringKey(SECRET.into())],
            steering_api_keys_file: Some(path),
        };
        let keys = resolve(
            &options,
            env(&[
                (KEYS_ENV, "env-one,env-two".into()),
                (KEYS_FILE_ENV, env_path.display().to_string()),
            ]),
        )
        .unwrap();
        assert_eq!(
            keys,
            ["flag-key", SECRET, "file-two", "file-three", "env-one", "env-two", "env-file-key"]
        );
    }

    #[test]
    fn an_empty_or_missing_key_file_fails_closed() {
        let empty = file("empty", "# nothing\n\n");
        for options in [
            SteeringAccessOptions { steering_api_keys_file: Some(empty.clone()), ..Default::default() },
            SteeringAccessOptions {
                steering_api_keys_file: Some(empty.with_file_name("missing")),
                ..Default::default()
            },
        ] {
            assert!(resolve(&options, env(&[])).is_err());
        }
        let from_env = [(KEYS_FILE_ENV, empty.display().to_string())];
        assert!(resolve(&SteeringAccessOptions::default(), env(&from_env)).is_err());
    }

    #[test]
    fn errors_and_debug_output_never_show_a_key() {
        let bad = format!("{SECRET} with-space");
        let path = file("bad", &format!("{SECRET}\nbad key\u{7}\n"));
        let failures = [
            resolve(
                &SteeringAccessOptions {
                    steering_api_key: vec![SteeringKey(bad.clone())],
                    ..Default::default()
                },
                env(&[]),
            ),
            resolve(&SteeringAccessOptions::default(), env(&[(KEYS_ENV, bad.clone())])),
            resolve(
                &SteeringAccessOptions { steering_api_keys_file: Some(path), ..Default::default() },
                env(&[]),
            ),
            resolve(
                &SteeringAccessOptions {
                    steering_api_key: vec![SteeringKey(" ".into())],
                    ..Default::default()
                },
                env(&[]),
            ),
        ];
        for failure in failures {
            let message = failure.unwrap_err();
            assert!(!message.contains(SECRET) && !message.contains("with-space"), "{message}");
        }
        let options = SteeringAccessOptions {
            steering_api_key: vec![SteeringKey(SECRET.into())],
            ..Default::default()
        };
        assert!(!format!("{options:?}").contains(SECRET));
    }
}
