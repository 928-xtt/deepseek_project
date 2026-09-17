class ContextManager:
    """最近 N 轮保留原文,更早的压成摘要拼进 system"""

    def __init__(self, client, keep_turns: int = 6, trigger_tokens: int = 3000):
        self.client = client
        self.keep_turns = keep_turns
        self.trigger_tokens = trigger_tokens
        self.summary = ""
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        total = sum(est_tokens(m["content"]) for m in self.history)
        if total > self.trigger_tokens:
            self._compress()

    def _compress(self):
        old = self.history[: -(self.keep_turns * 2)]
        self.summary = summarize(self.client, old, self.summary)
        self.history = self.history[-(self.keep_turns * 2) :]

    def build(self, system_prompt: str) -> list[dict]:
        msgs = [{"role": "system", "content": system_prompt}]
        if self.summary:
            msgs.append({"role": "system", "content": f"【历史摘要】{self.summary}"})
        return msgs + self.history
