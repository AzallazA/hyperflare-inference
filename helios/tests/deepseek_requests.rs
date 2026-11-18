/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(all(feature = "server", feature = "runtime-core"))]

#[path = "support/native_prefix.rs"]
mod native_prefix;

use flare_helios::{frontend, model, scheduler};
#[path = "support/frontend.rs"]
mod support;

use flare_helios::{
    frontend::{
        Asset, Frontend, FrontendAssets, FrontendSettings, PromptEncoder, TokenizerSettings,
    },
    grammar::GrammarCompiler,
    output::tagged::OutputStart,
    server::{PreparedRequest, prepare_request},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::{AddedToken, Tokenizer};

static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    frontend: Frontend,
    compiler: GrammarCompiler,
    config: model::ModelConfig,
}

fn tokenizer_asset() -> Asset {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(
            alphabet
                .into_iter()
                .enumerate()
                .map(|(id, c)| (c.to_string(), id as u32))
                .collect::<tokenizers::models::bpe::Vocab>(),
            vec![],
        )
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .add_special_tokens(
            [
                "<｜begin▁of▁sentence｜>",
                "<｜end▁of▁sentence｜>",
                "<｜System｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<think>",
                "</think>",
                "<｜deepseek_image｜>",
            ]
            .map(|s| AddedToken::from(s, true)),
        )
        .unwrap();
    tokenizer.add_tokens([AddedToken::from("｜DSML｜", false)]).unwrap();
    let bytes = tokenizer.to_string(false).unwrap();
    let path = std::env::temp_dir().join(format!(
        "flare-deepseek-request-{}-{}.json",
        std::process::id(),
        FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, &bytes).unwrap();
    Asset {
        path,
        sha256: format!("{:x}", Sha256::digest(bytes.as_bytes())),
        range: None,
    }
}

impl Fixture {
    fn new(encoder: PromptEncoder) -> Self {
        Self::with_defaults(encoder, json!({}))
    }

    fn with_defaults(encoder: PromptEncoder, defaults: Value) -> Self {
        let asset = tokenizer_asset();
        let template = (encoder == PromptEncoder::Jinja).then(|| {
            let text = "{% mark %}{{ messages | tojson }}\n{{ tools | default([]) | tojson }}\n{{ tool_choice | default('auto') }}";
            let path = asset.path.with_extension("jinja");
            std::fs::write(&path, text).unwrap();
            Asset { path, range: None, sha256: format!("{:x}", Sha256::digest(text.as_bytes())) }
        });
        let template_path = template.as_ref().map(|asset| asset.path.clone());
        let settings: FrontendSettings = serde_json::from_value(json!({
            "prompt_encoder": i32::from(encoder), "tokenizer": {}, "eos_token_ids": [257],
            "image_placeholder_token_id":Tokenizer::from_file(&asset.path).unwrap().token_to_id("<｜deepseek_image｜>").unwrap()
        }))
        .unwrap();
        let frontend = Frontend::load_with_settings(
            FrontendAssets {
                tokenizer: Asset {
                    path: asset.path.clone(),
                    sha256: asset.sha256.clone(),
                    range: asset.range,
                },
                tokens: TokenizerSettings::default(),
                chat_template: template,
            },
            defaults.as_object().unwrap().clone(),
            &settings,
        )
        .unwrap();
        std::fs::remove_file(asset.path).unwrap();
        if let Some(path) = template_path {
            std::fs::remove_file(path).unwrap();
        }
        let mut config =
            support::model_config(support::contract(if encoder == PromptEncoder::DeepseekV4 {
                "deepseek4"
            } else {
                "deepseek41"
            }));
        config.eos_token_ids = vec![
            frontend
                .tokenizer()
                .token_to_id("<｜end▁of▁sentence｜>")
                .unwrap(),
        ];
        let compiler = GrammarCompiler::new(
            frontend.tokenizer(),
            config.contract.vocabulary,
            &config.eos_token_ids,
            &flare_helios::model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend)
                .unwrap(),
        )
        .unwrap();
        Self {
            frontend,
            compiler,
            config,
        }
    }

    fn prepare(&self, extra: Value) -> flare_helios::Result<PreparedRequest> {
        let mut body = json!({"model": self.config.model_id,
            "messages": [{"role":"system","content":"Policy"},{"role":"user","content":"Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        prepare_request(&self.frontend, None, &self.compiler, &self.config, body)
    }

    fn accepts(&self, extra: Value, text: &str) -> bool {
        let mut request = self.prepare(extra).unwrap();
        let mut grammar = request.prompt.grammar.take().unwrap();
        self.frontend
            .encode(text, false)
            .unwrap()
            .into_iter()
            .chain(self.config.eos_token_ids.iter().copied())
            .all(|token| grammar.advance(token).is_ok())
    }
}

fn boundary_effort_options(encoder: PromptEncoder) -> Vec<Value> {
    let mut options = vec![json!({})];
    if encoder == PromptEncoder::DeepseekV41 {
        options.push(json!({"reasoning_effort":80}));
    }
    options
}

#[test]
fn named_selection_only_filters_tools_and_none_restores_a_text_turn() {
    for encoder in [PromptEncoder::Jinja, PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        let tools = json!([
            {"type":"function","function":{"name":"lookup","description":"Find a value",
                "parameters":{"type":"object","properties":{"query":{"type":"string"}}}}},
            {"type":"function","function":{"name":"other","description":"Another tool",
                "parameters":{"type":"object"}}}
        ]);
        let messages = json!([
            {"role":"system","content":"Policy"},
            {"role":"user","content":"Find a value"},
            {"role":"assistant","tool_calls":[{"id":"prior","type":"function","function":{"name":"other","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"prior","content":"Previous result"},
            {"role":"user","content":"Now look it up"}
        ]);
        // DeepSeek encoders: a named choice is "required" of the kept tool, available only with
        // thinking off; chat templates receive the kept tool with an automatic choice.
        let native = encoder != PromptEncoder::Jinja;
        let request = |tools: Value, choice: Value| {
            let mut body = json!({"messages":messages,"tools":tools,"tool_choice":choice});
            if native {
                body["enable_thinking"] = json!(false);
            }
            fixture.prepare(body).unwrap()
        };
        let named = request(tools.clone(), json!({"type":"function","function":{"name":"lookup"}}));
        let selected = request(json!([tools[0]]), json!(if native {"required"} else {"auto"}));
        assert_eq!(named.rendered_prompt, selected.rendered_prompt, "{encoder:?}");
        assert_eq!(named.prompt.tokens, selected.prompt.tokens);
        assert!(named.prompt.grammar.is_none());
        assert!(named.rendered_prompt.contains("Previous result"));
        assert_eq!(named.output.content.tools.len(), 1);
        assert_eq!(named.output.content.tools[0]["function"]["name"], "lookup");

        let followup = json!([
            {"role":"user","content":"Find a value"},
            {"role":"assistant","tool_calls":[{"id":"chosen","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"x\"}"}}]},
            {"role":"tool","tool_call_id":"chosen","content":"42"}
        ]);
        let none = fixture.prepare(json!({"messages":followup,"tools":tools,"tool_choice":"none"})).unwrap();
        let empty = fixture.prepare(json!({"messages":followup,"tools":[],"tool_choice":"none"})).unwrap();
        assert_eq!(none.rendered_prompt, empty.rendered_prompt);
        assert!(none.output.content.tools.is_empty());
        assert!(none.prompt.grammar.is_none());

        // Only a prompt that renders tool definitions counts as a tool request for the
        // scheduler's tools-only prefill tail merge.
        assert!(named.prompt.tools && selected.prompt.tools, "{encoder:?}");
        assert!(!none.prompt.tools && !empty.prompt.tools, "{encoder:?}");
        let plain = fixture.prepare(json!({"messages":[{"role":"user","content":"Hi"}]})).unwrap();
        assert!(!plain.prompt.tools, "{encoder:?}");
    }
}

#[test]
fn native_encoders_checkpoint_each_message_prefix() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        let messages = json!([
            {"role":"system","content":"Policy"},
            {"role":"assistant","content":"Opening reply"},
            {"role":"user","content":"Question"},
            {"role":"assistant","reasoning_content":"Thought", "content":"Answer"},
            {"role":"user","content":"Follow-up"}
        ]);
        for options in boundary_effort_options(encoder) {
            let prepared = fixture.prepare(json!({"messages":messages,"chat_template_kwargs":options})).unwrap();
            for count in 1..=messages.as_array().unwrap().len() {
                let prefix = native_prefix::without_generation(&fixture.frontend,
                    &messages.as_array().unwrap()[..count], options.as_object().unwrap());
                assert!(prepared.prompt.tokens.starts_with(&prefix.token_ids));
                assert!(prepared.prompt.message_boundaries.contains(&prefix.token_ids.len()),
                    "{encoder:?}, options {options}, message {count}");
            }
        }
    }
}

#[test]
fn native_boundaries_follow_merged_users_and_sorted_tool_results() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        let messages = json!([
            {"role":"system","content":"Policy"},
            {"role":"user","content":"First user"},
            {"role":"user","content":"Second user"},
            {"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"lookup","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":"Result B"},
            {"role":"tool","tool_call_id":"a","content":"Result A"}
        ]);
        for options in boundary_effort_options(encoder) {
            let prepared = fixture.prepare(json!({"messages":messages,"chat_template_kwargs":options})).unwrap();
            let text = &prepared.rendered_prompt;
            assert!(text.contains("First user\n\nSecond user"));
            assert!(text.find("Result A").unwrap() < text.find("Result B").unwrap());
            let token_end = |suffix: &str| {
                let byte = text.find(suffix).unwrap() + suffix.len();
                fixture.frontend.encode(&text[..byte], false).unwrap().len()
            };
            assert!(!prepared.prompt.message_boundaries.contains(&token_end("First user")));
            for suffix in ["Policy", "Second user", "Result A</tool_result>", "Result B</tool_result>"] {
                assert!(prepared.prompt.message_boundaries.contains(&token_end(suffix)),
                    "{encoder:?}, options {options}: {suffix}");
            }
        }
    }
}

#[test]
fn typed_native_settings_do_not_require_jinja_and_legacy_settings_do() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        assert_eq!(fixture.frontend.prompt_encoder(), encoder);
        let request = fixture.prepare(json!({})).unwrap();
        assert_eq!(request.output.content.start, OutputStart::Reasoning);
        assert!(request.rendered_prompt.ends_with("<｜Assistant｜><think>"));
        let boundary = *request.prompt.message_boundaries.first().unwrap();
        assert!(boundary > 0 && boundary < request.prompt.tokens.len());
    }
    let asset = tokenizer_asset();
    let settings: FrontendSettings =
        serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[257]})).unwrap();
    assert_eq!(settings.prompt_encoder, PromptEncoder::Jinja);
    let assets = FrontendAssets {
        tokenizer: Asset {
            path: asset.path.clone(),
            sha256: asset.sha256.clone(),
            range: asset.range,
        },
        tokens: TokenizerSettings::default(),
        chat_template: None,
    };
    assert!(Frontend::load_with_settings(assets, Map::new(), &settings).is_err());
    std::fs::remove_file(asset.path).unwrap();
    assert!(
        serde_json::from_value::<FrontendSettings>(
            json!({"prompt_encoder":3,"tokenizer":{},"eos_token_ids":[257]})
        )
        .is_err()
    );
}

#[test]
fn json_object_uses_the_official_prompt_and_schema_instructions_remain_explicit() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let object = json!({"enable_thinking":false,"response_format":{"type":"json_object"}});
        let actual = f.prepare(object.clone()).unwrap();
        let expected = f
            .frontend
            .render_chat(
                &[
                    json!({"role":"system","content":"Policy"}),
                    json!({"role":"user","content":"Hello"}),
                ],
                json!({"enable_thinking":false,"response_format":"json_object"})
                    .as_object()
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(actual.rendered_prompt, expected.text);
        assert_eq!(actual.prompt.tokens, expected.token_ids);
        assert_eq!(actual.output.content.start, OutputStart::Content);
        assert!(actual.prompt.grammar.is_some());
        assert!(f.accepts(object.clone(), "{\"ok\":true}"));
        assert!(!f.accepts(object, "[1]"));
        let schema = json!({"enable_thinking":false,"response_format":{"type":"json_schema","json_schema":{
            "name":"answer","strict":true,"schema":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}
        }}});
        let request = f.prepare(schema.clone()).unwrap();
        assert!(request.rendered_prompt.contains("JSON schema:"));
        assert!(f.accepts(schema.clone(), "{\"ok\":true}"));
        assert!(!f.accepts(schema, "{\"ok\":false}"));
    }
}

#[test]
fn required_tools_in_chat_mode_start_inside_the_dsml_envelope() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let options = json!({"enable_thinking":false,"tool_choice":"required",
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
            "response_format":{"type":"json_object"}});
        let request = f.prepare(options.clone()).unwrap();
        assert_eq!(
            request.output.content.start,
            OutputStart::ToolBody,
            "{}",
            request.rendered_prompt
        );
        let (open, invoke, close) = if encoder == PromptEncoder::DeepseekV4 {
            (
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"lookup\"></｜DSML｜invoke>",
                "</｜DSML｜tool_calls>",
            )
        } else {
            (
                "<｜DSML｜ calls>",
                "<｜DSML｜ invoke name=\"lookup\"></｜DSML｜ invoke>",
                "</｜DSML｜ calls>",
            )
        };
        assert!(request.rendered_prompt.trim_end().ends_with(open));
        assert!(request.prompt.grammar.is_some());
        assert!(!f.accepts(options.clone(), "{\"ok\":true}"));
        let completion = format!("{invoke}\n{close}{{\"ok\":true}}");
        assert!(f.accepts(options.clone(), &completion));
        assert!(!f.accepts(options.clone(), &format!("{invoke}{{\"ok\":true}}")));
        let mut automatic = options.clone();
        automatic["tool_choice"] = json!("auto");
        assert!(f.accepts(automatic, &format!("{open}{invoke}{close}{{\"ok\":true}}")));
        let mut output = flare_helios::output::OutputPipeline::new(request.output);
        let mut events = Vec::new();
        let mut decoder = f.frontend.incremental_decoder(false);
        for token in f.frontend.encode(&completion, false).unwrap() {
            events.extend(output.push(flare_helios::output::Input::Token(token)).unwrap());
            if let Some(text) = decoder.push(&[token]).unwrap() {
                events.extend(output.push(flare_helios::output::Input::Text(&text)).unwrap());
            }
        }
        let remaining = decoder.finish().unwrap();
        if !remaining.is_empty() {
            events.extend(output.push(flare_helios::output::Input::Text(&remaining)).unwrap());
        }
        events.extend(
            output
                .finish(flare_helios::output::Termination::Eos)
                .unwrap(),
        );
        assert!(events.iter().any(|event| matches!(event,
            flare_helios::output::Event::ToolBegin { name, .. } if name == "lookup")));
        assert!(
            events.contains(&flare_helios::output::Event::ToolArguments {
                index: 0,
                text: "{}".into()
            })
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, flare_helios::output::Event::Content(_)))
        );
    }
}

#[test]
fn native_reasoning_options_are_validated_through_request_preparation() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        for effort in ["low", "high", "xhigh", "max"] {
            assert!(f.prepare(json!({"reasoning_effort":effort})).is_ok());
        }
        for extra in [
            json!({"chat_template_kwargs":{"reasoning_effort":101}}),
            json!({"chat_template_kwargs":{"drop_thinking":true}}),
            json!({"chat_template_kwargs":{"continue_final_message":true}}),
            json!({"chat_template_kwargs":{"add_generation_prompt":false}}),
        ] {
            assert!(f.prepare(extra.clone()).is_err(), "{extra}");
        }
    }
}

fn assert_request_effort(fixture: &Fixture, request: &PreparedRequest, score: u8) {
    assert!(request.rendered_prompt.contains(&format!("Reasoning Effort: {score} (range 1-100")), "{}", request.rendered_prompt);
    assert_eq!(request.rendered_prompt.matches("Reasoning Effort:").count(), 1);
    assert_eq!(request.prompt.tokens,
        fixture.frontend.encode(&request.rendered_prompt, false).unwrap());
    assert_eq!(request.output.content.start, OutputStart::Reasoning);
}

#[test]
fn numeric_scores_reach_the_encoded_prompt_through_both_chat_option_paths() {
    let fixture = Fixture::new(PromptEncoder::DeepseekV41);
    for score in [1u8, 80, 100] {
        let direct = fixture.prepare(json!({"reasoning_effort":score})).unwrap();
        let kwargs = fixture.prepare(json!({"chat_template_kwargs":{"reasoning_effort":score}})).unwrap();
        assert_request_effort(&fixture, &direct, score);
        assert_eq!(direct.rendered_prompt, kwargs.rendered_prompt);
        assert_eq!(direct.prompt.tokens, kwargs.prompt.tokens);
    }
    let eighty = fixture.prepare(json!({"reasoning_effort":80})).unwrap();
    let high = fixture.prepare(json!({"reasoning_effort":"high"})).unwrap();
    assert_ne!(eighty.prompt.tokens, high.prompt.tokens);
    let nested_override = fixture.prepare(json!({"reasoning_effort":50,
        "chat_template_kwargs":{"reasoning_effort":80}})).unwrap();
    assert_eq!(nested_override.rendered_prompt, eighty.rendered_prompt);
    assert_eq!(nested_override.prompt.tokens, eighty.prompt.tokens);
    for (name, score) in [("low", 50), ("high", 75), ("xhigh", 75), ("max", 100)] {
        let named = fixture.prepare(json!({"reasoning_effort":name})).unwrap();
        let numeric = fixture.prepare(json!({"reasoning_effort":score})).unwrap();
        assert_eq!(numeric.rendered_prompt, named.rendered_prompt, "{name}");
        assert_eq!(numeric.prompt.tokens, named.prompt.tokens);
    }
}

#[test]
fn invalid_scores_and_v4_numeric_requests_fail_before_admission() {
    let invalid = [json!(0), json!(101), json!(-1), json!(80.0), json!(80.5),
        json!(true), json!(false), json!("80"), json!(u64::MAX),
        serde_json::from_str::<Value>("18446744073709551616").unwrap()];
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        for score in &invalid {
            for extra in [json!({"reasoning_effort":score}),
                          json!({"chat_template_kwargs":{"reasoning_effort":score}})] {
                assert!(fixture.prepare(extra.clone()).is_err(), "{encoder:?}: {extra}");
            }
        }
    }
    let v4 = Fixture::new(PromptEncoder::DeepseekV4);
    for score in [1, 80, 100] {
        assert!(v4.prepare(json!({"reasoning_effort":score})).is_err());
        assert!(v4.prepare(json!({"chat_template_kwargs":{"reasoning_effort":score}})).is_err());
    }
}

#[test]
fn numeric_defaults_null_and_named_requests_preserve_request_isolation() {
    let fixture = Fixture::with_defaults(PromptEncoder::DeepseekV41, json!({"reasoning_effort":80}));
    for (extra, score) in [
        (json!({}), 80),
        (json!({"reasoning_effort":1}), 1),
        (json!({"chat_template_kwargs":{"reasoning_effort":"low"}}), 50),
        (json!({"reasoning_effort":null}), 75),
        (json!({"chat_template_kwargs":{"reasoning_effort":null}}), 75),
        (json!({}), 80),
    ] {
        assert_request_effort(&fixture, &fixture.prepare(extra).unwrap(), score);
    }
    let named = Fixture::with_defaults(PromptEncoder::DeepseekV41, json!({"reasoning_effort":"low"}));
    assert_request_effort(&named, &named.prepare(json!({})).unwrap(), 50);
    let off = fixture.prepare(json!({"reasoning_effort":80,"enable_thinking":false})).unwrap();
    assert!(!off.rendered_prompt.contains("Reasoning Effort:"));
    assert!(off.rendered_prompt.ends_with("</think>"));
    assert_eq!(off.output.content.start, OutputStart::Content);
    assert_request_effort(&fixture, &fixture.prepare(json!({})).unwrap(), 80);
}

#[test]
fn base64_images_follow_the_rendered_placeholder_order() {
    use base64::Engine;
    use flare_helios::frontend::images::{
        DeepseekRowMajorImageSettings, ImageLimits, ImageProcessor, ImageSettings,
    };
    use flare_helios::model::contract::{
        ImageLayoutKind, ImagePreparationKind, ImageSpan, ImageTokenLayout,
    };

    fn encoded_image(rgb: [u8; 3]) -> (Vec<u8>, String) {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgb).unwrap();
        }
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );
        (bytes, url)
    }
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let mut f = Fixture::new(encoder);
        let settings = if encoder == PromptEncoder::DeepseekV41 {
            // This support fixture is text-only; declare the recipe's row-major image inputs locally.
            let image_start = f.config.contract.vocabulary as u32;
            f.config.contract.input_vocabulary = f.config.contract.vocabulary + 4;
            f.config.contract.vision = model::VisionCapabilities {
                feature_slots: 4,
                features_per_request: 4096,
                max_images: 4,
                patches_per_feature: 9,
                patch_capacities: vec![576, 1152, 2304, 4608, 9216],
            };
            let syntax = &mut f.config.contract.frontend;
            syntax.image_preparation = Some(ImagePreparationKind::DeepseekRowMajor);
            syntax.image_layout = Some(ImageLayoutKind::DeepseekRowMajor);
            syntax.rejected_tokens.clear();
            syntax.image_tokens = Some(ImageTokenLayout {
                feature_token: image_start + 1,
                span: Some(ImageSpan {
                    start_token: image_start,
                    end_token: image_start + 3,
                    max_tokens: 1024,
                    interior_tokens: vec![image_start + 2],
                    unscoped_tokens: vec![],
                }),
            });
            ImageSettings::DeepseekRowMajor(DeepseekRowMajorImageSettings {
                min_pixels: 46656,
                max_tokens: 1024,
                max_width_height_ratio: None,
            })
        } else {
            ImageSettings::Deepseek4 {}
        };
        let processor = ImageProcessor::load(settings, ImageLimits::default()).unwrap();
        for options in boundary_effort_options(encoder) {
            let (red, red_url) = encoded_image([255, 0, 0]);
            let (green, green_url) = encoded_image([0, 255, 0]);
            let messages = json!([{"role":"user","content":[
                {"type":"image_url","image_url":{"url":green_url}},
                {"type":"text","text":"First"},
                {"type":"image_url","image_url":{"url":red_url}},
                {"type":"text","text":"Second"}
            ]}]);
            let rendered = f
                .frontend
                .render_chat(messages.as_array().unwrap(), options.as_object().unwrap())
                .unwrap();
            assert_eq!(rendered.image_sources, vec![green_url, red_url]);
            let request = prepare_request(
                &f.frontend,
                Some(&processor),
                &f.compiler,
                &f.config,
                json!({"model":f.config.model_id,"messages":messages,"chat_template_kwargs":options}),
            )
            .unwrap();
            let expected = processor
                .process(&[
                    processor.decode_png(&green).unwrap(),
                    processor.decode_png(&red).unwrap(),
                ])
                .unwrap();
            let mut actual = flare_helios::frontend::images::ProcessedImages::default();
            for image in &request.prompt.conditioned_inputs {
                actual.append(image.materialize().unwrap());
            }
            assert_eq!(actual.pixel_values, expected.pixel_values);
            assert_eq!(actual.geometry.len(), 2);
            assert_eq!(request.rendered_prompt, rendered.text);
            if encoder == PromptEncoder::DeepseekV41 {
                let score = options["reasoning_effort"].as_u64().unwrap_or(75);
                assert!(request.rendered_prompt.contains(&format!("Reasoning Effort: {score} (range 1-100")), "{}", request.rendered_prompt);
            }
            let complete_message = native_prefix::without_generation(&f.frontend, messages.as_array().unwrap(), options.as_object().unwrap());
            let complete_message = flare_helios::model::prepare_input(&f.config, &f.frontend,
                Some(&processor), &complete_message,
                vec![processor.decode_png(&green).unwrap(), processor.decode_png(&red).unwrap()]).unwrap();
            assert!(request.prompt.tokens.starts_with(&complete_message.tokens));
            assert!(request.prompt.message_boundaries.contains(&complete_message.tokens.len()));
        }
    }
}

#[test]
fn decision_generation_prompt_prepares_a_nonthinking_first_assistant_token() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let mut fixture = Fixture::new(encoder);
        fixture.config.max_top_logprobs = 1;
        let prepared = fixture
            .prepare(json!({
                "messages":[
                    {"role":"system","content":"Select one label."},
                    {"role":"user","content":"State: charged twice"},
                    {"role":"assistant","content":"State received."},
                    {"role":"user","content":"Question: refund? A: false B: true\nAnswer:"}
                ],
                "chat_template_kwargs":{"enable_thinking":false,"add_generation_prompt":true},
                "max_tokens":1,"ignore_eos":true,"logprobs":true
            }))
            .unwrap();
        assert!(prepared.rendered_prompt.contains("Answer:"));
        assert!(prepared.rendered_prompt.ends_with("</think>"));
        assert_eq!(
            prepared.prompt.tokens,
            fixture
                .frontend
                .encode(&prepared.rendered_prompt, false)
                .unwrap()
        );
        assert_eq!(prepared.output.content.start, OutputStart::Content);
    }
}

#[test]
fn added_non_special_dsml_token_produces_tool_calls() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let token = f.frontend.tokenizer().token_to_id("｜DSML｜").unwrap();
        assert!(!f.frontend.tokenizer().get_added_vocabulary().is_special_token("｜DSML｜"));
        let request = f.prepare(json!({"tools":[{"type":"function","function":{
            "name":"get_weather","parameters":{"type":"object"}
        }}]})).unwrap();
        assert!(request.output.control_tokens.contains(&(token, "｜DSML｜".to_owned())));
        let (block, invoke, parameter) = if encoder == PromptEncoder::DeepseekV4 {
            ("tool_calls", "invoke", "parameter")
        } else {
            (" calls", " invoke", " parameter")
        };
        let completion = format!("</think>I'll check.\n\n<｜DSML｜{block}>\n<｜DSML｜{invoke} name=\"get_weather\">\n<｜DSML｜{parameter} name=\"location\" string=\"true\">Berlin</｜DSML｜{parameter}>\n</｜DSML｜{invoke}>\n</｜DSML｜{block}>");
        let mut output = flare_helios::output::OutputPipeline::new(request.output);
        let mut decoder = f.frontend.incremental_decoder(false);
        let mut events = Vec::new();
        for id in f.frontend.encode(&completion, false).unwrap() {
            events.extend(output.push(flare_helios::output::Input::Token(id)).unwrap());
            if let Some(text) = decoder.push(&[id]).unwrap() {
                events.extend(output.push(flare_helios::output::Input::Text(&text)).unwrap());
            }
        }
        let rest = decoder.finish().unwrap();
        if !rest.is_empty() {
            events.extend(output.push(flare_helios::output::Input::Text(&rest)).unwrap());
        }
        events.extend(output.finish(flare_helios::output::Termination::Eos).unwrap());
        assert!(events.iter().any(|event| matches!(event,
            flare_helios::output::Event::ToolBegin { name, .. } if name == "get_weather")));
        assert!(events.iter().any(|event| matches!(event,
            flare_helios::output::Event::ToolArguments { text, .. } if serde_json::from_str::<Value>(text).unwrap() == json!({"location":"Berlin"}))));
        assert!(events.iter().any(|event| matches!(event, flare_helios::output::Event::ToolEnd { .. })));
    }
}

#[test]
fn required_and_named_tools_are_rejected_in_thinking_mode() {
    let tools = json!([
        {"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},
        {"type":"function","function":{"name":"other","parameters":{"type":"object"}}}
    ]);
    let named = json!({"type":"function","function":{"name":"lookup"}});
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        // As the reference (deepseek-recipe convert_client_tools): 400 with its message, before any
        // generation. Request-level enable_thinking, template kwargs and the default all count.
        for thinking in [json!({}), json!({"enable_thinking":true}),
                         json!({"chat_template_kwargs":{"enable_thinking":true}}),
                         json!({"chat_template_kwargs":{"thinking_mode":"thinking"}})] {
            for choice in [json!("required"), named.clone()] {
                let mut body = thinking.clone();
                body["tools"] = tools.clone();
                body["tool_choice"] = choice.clone();
                let error = fixture.prepare(body).err().unwrap_or_else(|| panic!("{encoder:?} {thinking} {choice} accepted"));
                assert!(matches!(error, flare_helios::Error::Admission(flare_helios::AdmissionError::Invalid(_))), "{error:?}");
                assert!(error.to_string().contains("Thinking mode does not support this tool_choice"), "{error}");
            }
            let mut auto = thinking.clone();
            auto["tools"] = tools.clone();
            assert!(fixture.prepare(auto).is_ok());
        }
        // Thinking off still forces the call.
        for choice in [json!("required"), named.clone()] {
            let request = fixture.prepare(json!({"enable_thinking":false,"tools":tools,"tool_choice":choice})).unwrap();
            assert_eq!(request.output.content.start, OutputStart::ToolBody, "{encoder:?} {choice}");
        }
        assert!(fixture.prepare(json!({"tool_choice":"required","tools":[],"enable_thinking":false})).is_err());
    }
    // Chat templates (GLM-5.3 and the other Jinja models) keep accepting both in thinking mode.
    let template = Fixture::new(PromptEncoder::Jinja);
    for choice in [json!("required"), named] {
        assert!(template.prepare(json!({"enable_thinking":true,"tools":tools,"tool_choice":choice})).is_ok());
    }
}

/// Fork (TP3) prefill tail merge (`PrefillTailMerge::On`): the scheduler runs a chunk past the last
/// message's checkpoint only when the next turn re-renders the prompt end. DeepSeek V4.1 keeps the
/// `<think>` header of earlier assistant turns only when the prompt renders tools, so a thinking
/// turn without tools diverges right after its last message; `reasoning_prompt` marks exactly that
/// case, and the last message's prefix is shared in every case.
#[test]
fn reasoning_prompt_marks_turns_whose_next_turn_rewrites_the_generation_prompt() {
    let tools = json!([{"type":"function","function":{"name":"lookup","description":"Find a value",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]);
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        // (template kwargs, tools, reasoning_prompt, the next turn extends this whole prompt)
        for (kwargs, with_tools, reasoning, extends) in [
            (json!({}), false, true, false),
            (json!({"enable_thinking": true}), false, true, false),
            (json!({"enable_thinking": false}), false, false, true),
            (json!({"enable_thinking": true}), true, true, true),
            (json!({"enable_thinking": false}), true, false, true),
        ] {
            let first_messages = json!([{"role":"system","content":"Policy"},{"role":"user","content":"Hello"}]);
            let next_messages = json!([{"role":"system","content":"Policy"},{"role":"user","content":"Hello"},
                {"role":"assistant","content":"Hi there"},{"role":"user","content":"More please"}]);
            let request = |messages: &Value| {
                let mut body = json!({"messages": messages, "chat_template_kwargs": kwargs});
                if with_tools {
                    body["tools"] = tools.clone();
                }
                fixture.prepare(body).unwrap()
            };
            let (first, next) = (request(&first_messages), request(&next_messages));
            let case = format!("{encoder:?} {kwargs} tools={with_tools}");
            assert_eq!(first.prompt.tools, with_tools, "{case}");
            assert_eq!(first.prompt.reasoning_prompt, reasoning, "{case}");
            // DeepSeek V4 drops an absent reasoning block from history even with tools; V4.1 (DS41)
            // renders it empty, so its tool turns keep the `<think>` header.
            if encoder == PromptEncoder::DeepseekV41 || !with_tools {
                assert_eq!(next.prompt.tokens.starts_with(&first.prompt.tokens), extends, "{case}");
            }
            let last = *first.prompt.message_boundaries.last().unwrap();
            assert!(last < first.prompt.tokens.len(), "{case}");
            assert_eq!(next.prompt.tokens[..last], first.prompt.tokens[..last], "{case}");
        }
    }
}

/// Prompts rendered by the DeepSeek reference (deepseek-recipe 935817c through
/// recipe-compare/scripts/refharness) for the Chat body in each case's `request`.
fn assert_reference_prompts(prefix: &str) {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/deepseek41_reference_prompts.json")).unwrap();
    let fixture = Fixture::new(PromptEncoder::DeepseekV41);
    let mut checked = 0;
    for (name, case) in cases.as_object().unwrap() {
        if !name.starts_with(prefix) {
            continue;
        }
        let request = fixture
            .prepare(case["request"].clone())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(request.rendered_prompt, case["expected"].as_str().unwrap(), "{name}");
        assert_eq!(
            request.prompt.tokens,
            fixture.frontend.encode(&request.rendered_prompt, false).unwrap(),
            "{name}"
        );
        checked += 1;
    }
    assert!(checked > 0, "no reference cases named {prefix}*");
}

#[test]
fn developer_messages_render_as_reference_system_turns_in_place() {
    assert_reference_prompts("f3_");
}

#[test]
fn strict_tools_render_the_reference_normalized_schema() {
    assert_reference_prompts("f5_");
    let strict = json!({"type":"function","function":{"name":"set_flag","strict":true,
        "parameters":{"type":"object","properties":{"on":{"type":"boolean"}}}}});
    let plain = json!({"type":"function","function":{"name":"lookup",
        "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}});
    let native = Fixture::new(PromptEncoder::DeepseekV41);
    // Strict and non-strict tools may share a request (the reference rejects that mix only
    // because it then constrains generation, which Flare does not do).
    let mixed = native.prepare(json!({"tools":[strict, plain]})).unwrap();
    assert!(mixed.rendered_prompt.contains(r#"{"name": "set_flag", "description": "", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "additionalProperties": false, "required": ["on"]}}"#), "{}", mixed.rendered_prompt);
    assert!(mixed.rendered_prompt.contains(r#""parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}"#));
    // A schema the reference cannot normalize fails with the reference's message.
    let partial = json!({"type":"function","function":{"name":"set_flag","strict":true,
        "parameters":{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a"]}}});
    let error = native.prepare(json!({"tools":[partial]})).err().unwrap().to_string();
    assert!(error.contains("Required properties must match all properties in the object"), "{error}");
    // Chat templates keep rejecting strict schemas.
    let template = Fixture::new(PromptEncoder::Jinja);
    let error = template.prepare(json!({"tools":[strict]})).err().unwrap().to_string();
    assert!(error.contains("strict tool schemas require constrained tool generation"), "{error}");
    assert!(template.prepare(json!({"tools":[plain]})).is_ok());
}

/// Malformed tool-argument markup passes through as the DeepSeek reference returns it. The
/// expected arguments are the reference StreamProcessor's output for the same completions
/// (recipe-compare out/rust_parse.json, cases o09, o10, o12), except that a `string="false"`
/// value that is not JSON is retained as a JSON string (upstream #359), so that call is a
/// valid object and ends with ToolEnd.
#[test]
fn malformed_tool_arguments_pass_through_like_the_reference() {
    let f = Fixture::new(PromptEncoder::DeepseekV41);
    let call = |params: &str| {
        format!("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n{params}</｜DSML｜ invoke>\n</｜DSML｜ calls>")
    };
    for (completion, expected, raw) in [
        // string="false" with a value that is not JSON: retained as a string
        (call("<｜DSML｜ parameter name=\"city\" string=\"false\">Paris</｜DSML｜ parameter>\n"),
         r#"{"city": "Paris"}"#, false),
        // parameter without its string attribute
        (call("<｜DSML｜ parameter name=\"city\">Paris</｜DSML｜ parameter>\n"),
         "{\"city\">Paris</｜DSML｜: ", true),
        // duplicate parameter name
        (call("<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n"),
         r#"{"city": "Paris", "city": "Rome"}"#, true),
    ] {
        let request = f.prepare(json!({"enable_thinking":false,"messages":[{"role":"user","content":"x"}],
            "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]})).unwrap();
        let mut output = flare_helios::output::OutputPipeline::new(request.output);
        let mut decoder = f.frontend.incremental_decoder(false);
        let mut events = Vec::new();
        for id in f.frontend.encode(&completion, false).unwrap() {
            events.extend(output.push(flare_helios::output::Input::Token(id)).unwrap());
            if let Some(text) = decoder.push(&[id]).unwrap() {
                events.extend(output.push(flare_helios::output::Input::Text(&text)).unwrap());
            }
        }
        let rest = decoder.finish().unwrap();
        if !rest.is_empty() {
            events.extend(output.push(flare_helios::output::Input::Text(&rest)).unwrap());
        }
        events.extend(output.finish(flare_helios::output::Termination::Eos).unwrap());
        use flare_helios::output::Event;
        let calls: Vec<_> = events.iter().filter_map(|event| match event {
            Event::ToolArguments { text, .. } => Some(text.as_str()),
            _ => None,
        }).collect();
        assert_eq!(calls, [expected], "{completion}");
        assert!(events.iter().any(|event| matches!(event, Event::ToolBegin { name, .. } if name == "get_weather")));
        let end = if raw { Event::ToolEndRaw { index: 0 } } else { Event::ToolEnd { index: 0 } };
        assert!(events.contains(&end), "{events:?}");
        assert!(!events.iter().any(|event| matches!(event, Event::Content(text) if !text.trim().is_empty())));
    }
}

#[test]
fn reasoning_effort_spellings_follow_the_reference() {
    assert_reference_prompts("f7_");
    let fixture = Fixture::new(PromptEncoder::DeepseekV41);
    for (effort, score) in [("minimal", 50), ("medium", 75)] {
        for extra in [json!({"reasoning_effort":effort}),
                      json!({"chat_template_kwargs":{"reasoning_effort":effort}})] {
            assert_request_effort(&fixture, &fixture.prepare(extra).unwrap(), score);
        }
    }
    // `none` without a thinking setting is chat mode, even over a thinking default; an explicit
    // thinking setting wins and renders the default effort.
    let thinking_default = Fixture::with_defaults(PromptEncoder::DeepseekV41, json!({"enable_thinking":true}));
    for f in [&fixture, &thinking_default] {
        let none = f.prepare(json!({"reasoning_effort":"none"})).unwrap();
        assert!(!none.rendered_prompt.contains("Reasoning Effort:"));
        assert!(none.rendered_prompt.ends_with("<｜Assistant｜></think>"));
        assert_eq!(none.output.content.start, OutputStart::Content);
        for extra in [json!({"enable_thinking":true,"reasoning_effort":"none"}),
                      json!({"reasoning_effort":"none","chat_template_kwargs":{"thinking_mode":"thinking"}})] {
            assert_request_effort(f, &f.prepare(extra).unwrap(), 75);
        }
    }
    let v4 = Fixture::new(PromptEncoder::DeepseekV4);
    for effort in ["none", "minimal", "medium"] {
        assert!(v4.prepare(json!({"reasoning_effort":effort})).is_ok(), "{effort}");
    }
    assert!(fixture.prepare(json!({"reasoning_effort":"ultra"})).is_err());
}

#[test]
fn history_arguments_that_are_not_objects_render_as_the_reference_does() {
    assert_reference_prompts("f9_");
    let history = |arguments: &str| json!({"messages":[
        {"role":"user","content":"go"},
        {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":arguments}}]},
        {"role":"tool","tool_call_id":"c1","content":"err"}],
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]});
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let rendered = f.prepare(history("{city: Paris")).unwrap().rendered_prompt;
        assert!(rendered.contains(r#"name="arguments" string="true">{city: Paris</"#), "{encoder:?}: {rendered}");
    }
    // Chat templates still require an object.
    let template = Fixture::new(PromptEncoder::Jinja);
    for invalid in ["{city: Paris", "[1, 2]", "\"Paris\""] {
        assert!(template.prepare(history(invalid)).is_err(), "{invalid}");
    }
    assert!(template.prepare(history("{\"city\":\"Paris\"}")).is_ok());
}

#[test]
fn named_tool_choice_forces_the_call_in_chat_mode() {
    assert_reference_prompts("f10_");
    let tools = json!([
        {"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},
        {"type":"function","function":{"name":"other","parameters":{"type":"object"}}}
    ]);
    let named = json!({"type":"function","function":{"name":"lookup"}});
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let request = f.prepare(json!({"enable_thinking":false,"tools":tools,"tool_choice":named})).unwrap();
        let required = f.prepare(json!({"enable_thinking":false,"tools":[tools[0]],"tool_choice":"required"})).unwrap();
        assert_eq!(request.rendered_prompt, required.rendered_prompt, "{encoder:?}");
        assert_eq!(request.output.content.start, OutputStart::ToolBody);
        assert_eq!(request.output.content.tools.len(), 1);
    }
    // Chat templates keep receiving the filtered tools with an automatic choice.
    let template = Fixture::new(PromptEncoder::Jinja);
    let request = template.prepare(json!({"tools":tools,"tool_choice":named})).unwrap();
    assert!(request.rendered_prompt.trim_end().ends_with("auto"), "{}", request.rendered_prompt);
}
