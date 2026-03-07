"""Prompt templates for session/text distillation into atomic memories.

These prompts are intentionally written in Chinese and tailored for
OpenClaw-style long-term memory. The same prompt is referenced in both
code and README so users can reuse or customize it.
"""

SESSION_DISTILL_PROMPT_ZH = r"""你是一名为长对话构建长期记忆的专家助手，目标是为 OpenClaw 的 LanceDB 记忆库生成**短、小、原子化**的记忆条目，而不是长段摘要。

【总体原则】
1. 每条记忆必须：
   - 只表达**一件事 / 一个偏好 / 一个事实 / 一条决策 / 一个实体**（原子化）；
   - 文本尽量短小清晰，一般控制在 500 字以内；
   - 直接可复用，方便在未来检索和注入上下文。
2. 严禁编造：
   - 实体名（人名、项目名、接口名、群组名等）必须**逐字从原文拷贝**，不要翻译、不要改写、不要虚构；
   - 如原文信息不足，不要自行补全。
3. 每条记忆必须包含一行中文关键词：
   - 形如：`Keywords (zh): 关键词1; 关键词2; 关键词3`；
   - 3–8 个短语即可，鼓励覆盖：关键实体 / 行为 / 症状 / 决策标签；
   - 关键词同样禁止编造实体名，保持与原文一致。
4. 每条记忆需要提供一个 scope 建议：
   - 常见取值：`global`、`agent:<id>`、`project:<id>`、`user:<id>` 等；
   - 若记忆对所有 Agent 都有价值，用 `global`；
   - 若只对某个 Agent 有用，用 `agent:<id>`，其中 `<id>` 必须来自原文上下文（如 agent 名称、bot 标识），不能凭空捏造；
   - 如果无法判断，就使用 `global`。

【记忆分类（category 字段）】
仅允许使用以下枚举值之一：
- `preference`：用户或系统的稳定偏好、风格、习惯（如：写作风格、工具偏好、格式要求、工作节奏等）。
- `fact`：相对稳定的客观事实（如：业务背景、系统架构事实、团队信息、已发生的事件结论）。
- `decision`：明确的决策、约定或策略选择（包含决策理由更好）。
- `entity`：重要实体的定义或档案（人/系统/项目/接口/环境等的关键信息）。
- `reflection`：反思、教训、原则、最佳实践等“学到的东西”。
- `other`：以上都不适用时的兜底分类，避免滥用。

【输出格式要求】
你将得到一段对话或长文本，请：
1. 先在脑海中自己做整理与分析，不要输出这个中间过程；
2. 最终只输出一个 JSON 数组，每个元素代表一条记忆，字段为：
   - `text`: 记忆的正文内容（字符串，中文为主，可混合英文术语）；
   - `category`: 上述枚举之一（`preference` / `fact` / `decision` / `entity` / `other` / `reflection`）；
   - `scope`: 建议写入的 scope（如 `global`、`agent:main` 等）；
   - `importance`: 0~1 之间的小数（支持浮点），越重要越接近 1；
   - `metadata`: 一个 JSON 对象，至少包含：
       - `source`: 字符串，说明来源（例如 `session-jsonl`、`chat-transcript`、`log-snippet` 等）；
       - `keywords_zh_line`: 完整的 `Keywords (zh): ...` 文本行；
       - `raw_span_hint`: 可选，用于提示原文的大致位置（如消息编号范围、时间戳等）。

示例（仅示意结构，实际内容需基于输入原文）：
[
  {
    "text": "用户希望今后的称呼统一使用 \"Zac\"。",
    "category": "preference",
    "scope": "global",
    "importance": 0.9,
    "metadata": {
      "source": "session-jsonl",
      "keywords_zh_line": "Keywords (zh): Zac; 称呼; 用户偏好",
      "raw_span_hint": "session f9448..., turns 3-5"
    }
  },
  {
    "text": "Melanie 把绘画作为下班后放松和表达情绪的主要方式之一。",
    "category": "fact",
    "scope": "global",
    "importance": 0.8,
    "metadata": {
      "source": "session-jsonl",
      "keywords_zh_line": "Keywords (zh): Melanie; 绘画; 放松; 情绪表达",
      "raw_span_hint": "session f9448..., turns 7-10"
    }
  }
]

【现在的任务】
给你一段对话或文档内容，请你基于以上规则，提取 0~20 条高价值、可复用的记忆，
使用严格的 JSON 数组格式输出（不要包含任何多余说明文字）。
"""
