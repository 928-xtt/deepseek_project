import tiktoken

from src.api_client import DeepSeekClient



"""
self.history 里永远只有“最近 N 轮原文”，不包含 system 提示词，
也不包含摘要——这两样是 build 时动态拼上去的。
所以 history 本身从来不是完整的请求体，必须过一遍 build。

"""


class ContextManager:
    """
    对话上下文管理器:控制发送给 LLM 的历史长度,防止 token 爆炸。

    策略:最近 N 轮保留原文,更早的对话压成一段摘要,拼在 system 消息里。
    """

    def __init__(self, client, keep_turns: int = 6, trigger_tokens: int = 3000):
        """
        构造方法(Python 的 __init__ 相当于 Java 的构造器,self 相当于 this)

        参数:
            client:         OpenAI 风格的客户端实例(DeepSeek 也用它),
                            _compress 里调用 summarize 时需要,所以存下来
            keep_turns:     压缩时保留最近几轮"原文"(一轮 = user + assistant 各一条)
            trigger_tokens: 历史总 token 超过这个数就触发压缩(警戒线)
        """
        self.client = client
        self.keep_turns = keep_turns
        self.trigger_tokens = trigger_tokens
        self.summary = ""  # 累积摘要缓存,初始为空(还没压缩过)
        self.history: list[dict] = (
            []
        )  # 原始历史消息,每条形如 {"role": "user", "content": "..."}

    def add(self, role: str, content: str):
        """
        新增一条对话消息,并检查是否需要触发压缩。

        调用时机:每收到一条用户消息/生成一条助手回复,就调一次。
        """
        # ① 新消息先入历史列表
        self.history.append({"role": role, "content": content})

        # ② 先判断该不该压,再压:把"判断条件"和"执行动作"拆开,
        #    条件自己一个方法,以后想单测阈值逻辑不用真的去调模型
        if self._should_compress():
            self._compress()

    def _should_compress(self) -> bool:
        """
        该不该触发压缩?两个条件必须同时成立:

        1. 历史总 token 超过警戒线(等 usage 告诉你超了就晚了:钱已花、请求可能已失败);
        2. 确实存在"更早的消息"可以压,也就是条数多于要保留的 keep_turns*2 条。

        第 2 条是这次修的坑。原来只看 token:单条消息就能超警戒线
        (比如粘贴一整段 4400 token 的代码),那时 old 切片是空的 ——
        摘要模型收到的是"对话内容:"后面什么都没有,白花一次调用,
        还会把"没有内容可总结"这种废话写回 self.summary。
        """
        # ① 没有老消息可压时,压完历史也不会变短,纯属浪费:直接不压
        if len(self.history) <= self.keep_turns * 2:
            return False

        # ② 估算整个历史的总 token 数(生成器表达式:逐条算,求和)
        total = sum(est_tokens(m["content"]) for m in self.history)
        return total > self.trigger_tokens

    # 只有当token超限制才调用
    # 顺带更新新的摘要和history
    def _compress(self):
        """
        执行压缩:老消息压成摘要并入缓存,history 只留最近 keep_turns 轮。

        方法名前的下划线是 Python 惯例:表示"内部方法,外部别直接调"。
        相当于 Java 里把方法声明为 private 的意图(但 Python 只是君子协定,不强制)。
        """
        # 一轮 = user + assistant 两条消息,所以保留条数是 keep_turns * 2
        keep = self.keep_turns * 2

        # ① 切出"老消息"切片:从头到 倒数第 keep 条 之前
        old = self.history[:-keep]

        # ② 兜底:老消息为空就别去调摘要模型了
        #    (正常路径已经被 _should_compress 挡住,这里是防御性写法:
        #     万一以后有人直接调 _compress(),也不会产生一次空摘要调用)
        if not old:
            return

        # ③ 把老消息交给 LLM 压缩,和上一次的摘要合并(增量滚动摘要)
        #    prev_summary 传入 self.summary,让 LLM 在旧摘要基础上更新,
        #    而不是每次从零总结全部历史
        self.summary = summarize(self.client, old, self.summary)

        # ④ history 重新赋值:只保留最近 keep 条原文
        self.history = self.history[-keep:]

    def build(self, system_prompt: str) -> list[dict]:
        """
        组装最终发送给 API 的完整 messages 列表。

        结构:
            [system 系统提示词,
             system 历史摘要(如果有的话),   ← 关键:摘要以 system 身份注入
             user 第1条, assistant 第1条,   ← 最近 N 轮原文
             ...
             user 最新一条, assistant 最新一条]

        调用时机:每次发请求前调用,返回值直接传给 client.chat.completions.create(messages=...)
        """
        # 列表里先放系统提示词(列表推导式之外的最基础写法)
        msgs = [{"role": "system", "content": system_prompt}]

        if self.summary:
            msgs.append({"role": "system", "content": f"【历史摘要】{self.summary}"})

        # 最后拼上最近 N 轮原文,一次性返回
        # (列表 + 列表 = 新列表,不会改动 self.history 本身)
        return msgs + self.history

# 把旧对话交给 DeepSeek 压成一段摘要
def summarize(
            client: DeepSeekClient, old_messages: list[dict] = [], prev_summary: str = ""
        ) -> str:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
            prompt = f"""请把以下对话压缩到 300 字内,必须保留:
        1. 用户核心目标  2. 已确认事实  3. 已做决策  4. 待办事项
        {"前情摘要:" + prev_summary if prev_summary else ""}
        对话内容:
        {text}"""
    
            resp = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            print(f"压缩后的摘要{resp}")
            return resp

# 在调用llm之前,计算本次对话输入消耗的token(包含了中文和英文两种语言)
def est_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")  # DeepSeek 与 OpenAI tokenizer 兼容
    return len(enc.encode(text))

