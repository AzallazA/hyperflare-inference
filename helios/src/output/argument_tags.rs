/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! GLM think envelopes and raw-string/JSON argument tags from the chat template.

use crate::{
    Result,
    output::tagged::{RawArguments, ToolDialect, parameter_value, strict_nonstring_raw},
};
use serde_json::Value;

#[cfg(test)]
use crate::output::tagged::{OutputConfig, TaggedOutput};

/// The argument tags that `ArgumentTags::parse` reads. They are markers of the parser that
/// `ToolDialect` does not carry; `OutputPipeline::declared_markers` lists them.
pub(crate) const ARGUMENT_MARKERS: [&str; 4] =
    ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];

pub struct ArgumentTags;
impl ToolDialect for ArgumentTags {
    const NAME: &'static str = "GLM";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        Self::parse_candidate(body, tools)?.ok_or_else(|| "incomplete JSON argument".into())
    }
    fn parse_candidate(body: &str, tools: &[Value]) -> Result<Option<Vec<(String, String)>>> {
        let body = body.trim();
        let split = body.find("<arg_key>").unwrap_or(body.len());
        let name = body[..split].trim();
        if name.is_empty() {
            return Err("tool function name is empty".into());
        }
        let mut rest = &body[split..];
        let function = tools
            .iter()
            .filter_map(|tool| tool.get("function"))
            .find(|function| function.get("name").and_then(Value::as_str) == Some(name));
        let strict = function.is_some_and(|f| f["strict"].as_bool() == Some(true));
        let properties = function
            .and_then(|function| function.get("parameters"))
            .and_then(|parameters| parameters.get("properties"))
            .and_then(Value::as_object);
        let mut arguments = RawArguments::new();
        while !rest.trim().is_empty() {
            let remaining = rest.trim_start();
            let key = remaining.strip_prefix("<arg_key>").ok_or_else(|| {
                // Name what follows instead of echoing generated text into the error.
                let marker = if remaining.starts_with("</arg_value>") {
                    "closing-arg-value"
                } else if remaining.starts_with("</tool_call>") {
                    "closing-tool-call"
                } else if remaining.starts_with("<arg_value>") {
                    "arg-value-without-key"
                } else if remaining.starts_with("<arg_key") {
                    "malformed-arg-key"
                } else if remaining.starts_with('<') {
                    "unexpected-tag"
                } else {
                    "unexpected-text"
                };
                format!(
                    "GLM tool call requires an argument key (remaining_bytes={}, next_marker={marker})",
                    remaining.len()
                )
            })?;
            let (key, tail) = key
                .split_once("</arg_key>")
                .ok_or("unterminated GLM argument key")?;
            // Keys are trimmed as vLLM's GLM-4.7 parser does (`key.strip()`); values stay raw.
            let key = key.trim();
            if key.is_empty() || arguments.contains_key(key) {
                return Err("GLM argument names must be nonempty and unique".into());
            }
            let value = tail
                .trim_start()
                .strip_prefix("<arg_value>")
                .ok_or("GLM argument requires a value tag")?;
            let schema = properties.and_then(|p| p.get(key));
            let Ok(container) =
                super::tagged::container_parameter(value, schema.filter(|_| strict))
            else {
                return Ok(None);
            };
            if let Some((parsed, remaining)) = container
                && let Some(next) = remaining.strip_prefix("</arg_value>")
            {
                arguments.insert(key.to_owned(), parsed);
                rest = next;
                continue;
            }
            let end = value_end(value).ok_or("unterminated GLM argument value")?;
            let (value, tail) = (&value[..end], &value[end + VALUE_END.len()..]);
            if let Some(raw_json) = strict_nonstring_raw(value, schema.filter(|_| strict)) {
                arguments.insert(key.to_owned(), raw_json);
            } else {
                // Off-schema values are retained (parameter_value); only an invalid type declaration fails here.
                let converted = parameter_value(value, schema).map_err(|error| {
                    let declared = schema
                        .and_then(|schema| schema.get("type"))
                        .map_or_else(|| "<missing-or-union>".into(), Value::to_string);
                    crate::FrontendError::Input(crate::Failure::caused_by(
                        format!(
                            "GLM argument '{key}' (declared type {declared}) failed conversion: {error}"
                        ),
                        error,
                    ))
                })?;
                arguments.insert_value(key.to_owned(), &converted)?;
            }
            rest = tail;
        }
        Ok(Some(vec![(name.to_owned(), arguments.to_json_string()?)]))
    }
    fn starts_call(text: &str) -> Option<bool> {
        // Calls continue with the function name and an argument key or closing tag;
        // quoted tags followed by prose remain text.
        let text = text.strip_prefix(Self::OPEN).unwrap_or(text).trim_start();
        if text.is_empty() {
            return None;
        }
        let end = text.find(|c: char| c == '<' || c == '>' || c.is_whitespace())?;
        if end == 0 {
            return Some(false);
        }
        let next = text[end..].trim_start();
        if ["<arg_key>", "</tool_call>"]
            .iter()
            .any(|tag| next.starts_with(tag))
        {
            Some(true)
        } else if ["<arg_key>", "</tool_call>"]
            .iter()
            .any(|tag| tag.starts_with(next))
        {
            None
        } else {
            Some(false)
        }
    }
}

const VALUE_END: &str = "</arg_value>";

/// Byte offset of the `</arg_value>` that closes a value, in the text after `<arg_value>`.
///
/// Values are raw text, so code or XML passed as a string can contain `</arg_value>` itself
/// (vLLM's non-greedy regex splits there). A closing tag ends the value only when, after optional
/// whitespace, the next argument key or the end of the call body follows; the body handed to
/// `parse` stops before `</tool_call>`, so that tag counts as the end. Any other closing tag is
/// value text. A value that literally contains `</arg_value><arg_key>` stays ambiguous and ends
/// there. When no closing tag qualifies, the first one ends the value as before, so a malformed
/// remainder still gets the next-marker diagnostic. Whole calls are parsed only after their
/// closing tag arrives, so streaming never emits a value prefix that a later tag could extend.
fn value_end(value: &str) -> Option<usize> {
    let mut first = None;
    for (at, _) in value.match_indices(VALUE_END) {
        first.get_or_insert(at);
        let next = value[at + VALUE_END.len()..].trim_start();
        if next.is_empty() || next.starts_with("<arg_key>") {
            return Some(at);
        }
    }
    first
}

#[cfg(test)]
type ArgumentTagOutput = TaggedOutput<ArgumentTags>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        output::{Input, ModelOutput, Termination},
        server::output::Accumulator,
    };
    use serde_json::json;

    fn config() -> OutputConfig {
        OutputConfig {
            structured: false,
            start: crate::output::tagged::OutputStart::Reasoning,
            tools: vec![json!({"type":"function", "function": {
            "name":"run", "parameters":{"type":"object", "properties":{
                "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
                "items":{"type":"array"}, "options":{"type":"object"}, "empty":{"type":"null"}
            }}}})],
        }
    }

    #[test]
    fn declared_markers_are_the_tags_the_parser_reads() {
        let [key, key_end, value, value_end] = ARGUMENT_MARKERS;
        let parsed = ArgumentTags::parse(
            &format!("run{key}text{key_end}{value}hello{value_end}"),
            &config().tools,
        )
        .unwrap();
        assert_eq!(
            parsed,
            [("run".to_owned(), r#"{"text":"hello"}"#.to_owned())]
        );
        let syntax = serde_json::from_str::<crate::model::PreparedContract>(include_str!(
            "../../tests/support/contracts/glm53.json"
        ))
        .unwrap()
        .frontend;
        let declared = crate::output::OutputPipeline::declared_markers(&syntax);
        assert!(
            ARGUMENT_MARKERS.iter().all(|tag| declared.contains(tag)),
            "{declared:?}"
        );
    }

    #[test]
    fn split_streams_preserve_reasoning_and_decode_typed_tool_arguments() {
        let text = "\nreason α\n</think>\nAnswer\n<tool_call>run<arg_key>text</arg_key><arg_value>\nhello α\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value><arg_key>options</arg_key><arg_value>{\"x\":1}</arg_value><arg_key>empty</arg_key><arg_value>null</arg_value></tool_call><tool_call>run</tool_call>\nend";
        for boundary in text
            .char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
        {
            let mut parser = ArgumentTagOutput::new(config());
            let mut output = Accumulator::new("test".into());
            output
                .apply(parser.push(Input::Text(&text[..boundary])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[boundary..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "\nreason α\n");
            assert_eq!(message["content"], "\nAnswer\n\nend");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0]["function"]["name"], "run");
            let arguments: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(
                arguments,
                json!({"text":"\nhello α\n", "count":2, "enabled":true, "items":[1,2], "options":{"x":1}, "empty":null})
            );
            assert_eq!(calls[1]["function"]["arguments"], "{}");
        }
    }

    #[test]
    fn malformed_calls_fail() {
        for body in [
            "",
            "run<arg_key></arg_key><arg_value>x</arg_value>",
            "run<arg_key>text</arg_key><arg_value>a</arg_value><arg_key>text</arg_key><arg_value>b</arg_value>",
            "run<arg_key>text</arg_key><arg_value>unfinished",
            "run<arg_key>x</arg_key>text",
        ] {
            assert!(
                ArgumentTags::parse(body, &config().tools).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn invalid_type_declarations_fail_unless_a_valid_branch_matches() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "bad":{"type":42}, "union":{"anyOf":[{"type":42}]},
            "mixed":{"oneOf":[{"type":42},{"type":"integer"}]}
        }}}})];
        let call = |key: &str, value: &str| {
            ArgumentTags::parse(
                &format!("run<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"),
                &tools,
            )
        };
        for (key, value) in [("bad", "123"), ("union", "123"), ("mixed", "two")] {
            assert!(call(key, value).is_err(), "{key}={value}");
        }
        assert_eq!(call("mixed", "123").unwrap()[0].1, "{\"mixed\":123}");
    }

    #[test]
    fn mismatched_argument_types_retain_generated_values_without_coercion() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "count":{"type":"integer"}, "enabled":{"type":"boolean"},
            "items":{"type":"array"}, "choice":{"anyOf":[{"type":"integer"},{"type":"null"}]},
            "listed":{"anyOf":[{"type":"object"},{"type":"string"}]}
        }}}})];
        let text = "</think><tool_call>run<arg_key>enabled</arg_key><arg_value>\"true\"</arg_value><arg_key>count</arg_key><arg_value>two</arg_value><arg_key>items</arg_key><arg_value>2.5</arg_value><arg_key>choice</arg_key><arg_value>false</arg_value><arg_key>listed</arg_key><arg_value>3</arg_value></tool_call>";
        for split in text.char_indices().map(|(at, _)| at) {
            let mut parser = ArgumentTagOutput::new(OutputConfig {
                tools: tools.clone(),
                ..config()
            });
            let mut output = Accumulator::new("test".into());
            for part in [&text[..split], &text[split..]] {
                output
                    .apply(parser.push(Input::Text(part)).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            let arguments: Value = serde_json::from_str(
                message["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                arguments,
                json!({"enabled":"true", "count":"two", "items":2.5, "choice":false, "listed":"3"}),
                "split {split}"
            );
        }
    }

    #[test]
    fn prose_tool_tag_mentions_remain_answer_text() {
        let call = "<tool_call>run<arg_key>text</arg_key><arg_value>x</arg_value></tool_call>";
        for (text, content, calls) in [
            (
                "</think>The `<tool_call>` tag wraps calls. END".to_owned(),
                "The `<tool_call>` tag wraps calls. END",
                0,
            ),
            (
                "</think>Write <tool_call> or `<tool_call>run` literally.".to_owned(),
                "Write <tool_call> or `<tool_call>run` literally.",
                0,
            ),
            (
                format!("</think>Quote `<tool_call>` here.\n{call}"),
                "Quote `<tool_call>` here.\n",
                1,
            ),
        ] {
            for split in text.char_indices().map(|(at, _)| at) {
                let mut parser = ArgumentTagOutput::new(config());
                let mut output = Accumulator::new("test".into());
                for part in [&text[..split], &text[split..]] {
                    output
                        .apply(parser.push(Input::Text(part)).unwrap())
                        .unwrap();
                }
                output
                    .apply(parser.finish(Termination::Eos).unwrap())
                    .unwrap();
                output.finish().unwrap();
                let message = output.message();
                assert_eq!(message["content"], content, "{text:?} at {split}");
                let tools = message["tool_calls"]
                    .as_array()
                    .map_or(&[][..], Vec::as_slice);
                assert_eq!(tools.len(), calls, "{text:?} at {split}");
                if let Some(tool) = tools.first() {
                    assert_eq!(tool["function"]["name"], "run");
                    assert_eq!(tool["function"]["arguments"], "{\"text\":\"x\"}");
                }
            }
        }
    }

    #[test]
    fn truncation_and_structured_output_keep_protocol_semantics() {
        let mut parser = ArgumentTagOutput::new(config());
        parser
            .push(Input::Text("</think><tool_call>run<arg_key>text</arg_key>"))
            .unwrap();
        assert!(parser.finish(Termination::OutputLimit).unwrap().is_empty());
        let mut parser = ArgumentTagOutput::new(OutputConfig {
            structured: true,
            start: crate::output::tagged::OutputStart::Content,
            ..config()
        });
        assert_eq!(
            parser
                .push(Input::Text("<think>{\"x\":1}</think>"))
                .unwrap(),
            vec![crate::output::Event::Content(
                "<think>{\"x\":1}</think>".into()
            )]
        );
    }
}
