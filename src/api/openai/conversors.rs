use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestDeveloperMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart,
};

#[derive(Debug, Clone)]
pub struct HfMsg {
    pub role: String,
    pub content: String,
}
pub fn to_hf_msg(m: &ChatCompletionRequestMessage) -> HfMsg {
    match m {
        ChatCompletionRequestMessage::Developer(x) => HfMsg {
            role: "system".into(), // HF normalmente no usa "developer"
            content: dev_content_to_string(&x.content),
        },
        ChatCompletionRequestMessage::System(x) => HfMsg {
            role: "system".into(),
            content: sys_content_to_string(&x.content),
        },
        ChatCompletionRequestMessage::User(x) => HfMsg {
            role: "user".into(),
            content: user_content_to_string(&x.content),
        },
        ChatCompletionRequestMessage::Assistant(x) => HfMsg {
            role: "assistant".into(),
            content: x
                .content
                .as_ref()
                .map(asst_content_to_string)
                .unwrap_or_default(),
        },
        ChatCompletionRequestMessage::Tool(x) => HfMsg {
            role: "tool".into(), // o "assistant" si tu template no soporta "tool"
            content: tool_content_to_string(&x.content),
        },
        ChatCompletionRequestMessage::Function(x) => HfMsg {
            role: "tool".into(), // legacy -> tool
            content: x.content.clone().unwrap_or_default(),
        },
    }
}

pub fn dev_content_to_string(c: &ChatCompletionRequestDeveloperMessageContent) -> String {
    match c {
        ChatCompletionRequestDeveloperMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestDeveloperMessageContent::Array(parts) => parts
            .iter()
            .map(|p| match p {
                ChatCompletionRequestDeveloperMessageContentPart::Text(t) => t.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

pub fn sys_content_to_string(c: &ChatCompletionRequestSystemMessageContent) -> String {
    match c {
        ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestSystemMessageContent::Array(parts) => parts
            .iter()
            .map(|p| match p {
                ChatCompletionRequestSystemMessageContentPart::Text(t) => t.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

pub fn user_content_to_string(c: &ChatCompletionRequestUserMessageContent) -> String {
    match c {
        ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestUserMessageContent::Array(parts) => parts
            .iter()
            .map(|p| match p {
                ChatCompletionRequestUserMessageContentPart::Text(t) => t.text.clone(),
                _ => "".to_string(), // imagen/audio/file: ignora o usa marcador
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

pub fn asst_content_to_string(c: &ChatCompletionRequestAssistantMessageContent) -> String {
    match c {
        ChatCompletionRequestAssistantMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestAssistantMessageContent::Array(parts) => parts
            .iter()
            .map(|p| match p {
                ChatCompletionRequestAssistantMessageContentPart::Text(t) => t.text.clone(),
                ChatCompletionRequestAssistantMessageContentPart::Refusal(r) => r.refusal.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

pub fn tool_content_to_string(c: &ChatCompletionRequestToolMessageContent) -> String {
    match c {
        ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
        ChatCompletionRequestToolMessageContent::Array(parts) => parts
            .iter()
            .map(|p| match p {
                ChatCompletionRequestToolMessageContentPart::Text(t) => t.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}
