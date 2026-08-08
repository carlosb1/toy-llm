use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};

pub const GENERIC_CHAT_TEMPLATE: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}
{% endfor %}{% if add_generation_prompt %}assistant:
{% endif %}"#;

pub fn load_chat_template(tokenizer_config_path: Option<String>) -> anyhow::Result<ChatTemplate> {
    let template = match tokenizer_config_path {
        Some(path) => {
            let json = std::fs::read_to_string(path)?;
            let config: TokenizerConfig = serde_json::from_str(&json)?;
            let template = ChatTemplate::from_tokenizer_config(&config)?;
            template
        }
        None => ChatTemplate::from_str(GENERIC_CHAT_TEMPLATE)?,
    };
    Ok(template)
}
pub struct PromptProcessor {
    chat_template: ChatTemplate,
}

impl PromptProcessor {
    pub fn new(chat_template: ChatTemplate) -> Self {
        PromptProcessor { chat_template }
    }
    pub fn encode_chat(&self, messages: &[Message]) -> anyhow::Result<String> {
        let prompt = self
            .chat_template
            .render_messages(&messages, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(prompt)
    }
}
