"""客户端单元测试：全程离线，不消耗任何 token，不花一分钱。

运行方式（在项目根目录下）：
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

【测试为什么可以不花钱】
DeepSeekClient 真正"出门打电话"的地方，从头到尾只有一行：

    response = self.client.chat.completions.create(**params)

这行一旦执行，就是在往 https://api.deepseek.com 发 HTTP 请求，
服务端按 input/output 的 token 数计费 —— 这才是花钱的那一步。
测试要做的，是把它换成一个"看着像、但只是在记账"的替身（mock）：
替身不发网络请求，只负责回答"你刚才用什么参数调了我"，并返回我们预先备好的假响应。
没有 HTTP 请求 → 没有 token → 没有账单。

【替身是从哪个口子塞进去的】
就是 api_client.py 里 __init__ 的 client 参数（见那边的注释）：
注入 client 时不再 load_dotenv()、不再 new OpenAI()，配置权完全交给调用方。
这个参数存在的一半理由，就是为了让下面这些测试成为可能。

【额外收获，比"省钱"更重要】
真 API 你没法命令它"这一次必须超时""这一次必须返回 429"。
而 mock 可以让 429 按需出现 —— 于是重试逻辑、降级文案、401 不重试这类
平时根本造不出来的分支，第一次变得可测、可回归。
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest import main as unittest_main
from unittest.mock import MagicMock, patch

# 让测试无论从哪个目录被运行，都能 import 到 src / models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # 本环境的 httpx 安装名被镜像成了 httpx2，两种写法都兼容
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from models.ResumeInfo import ResumeInfo
from src.api_client import DeepSeekClient
from src.jisoModeAndPydantic import extract_with_retry


# --------------------------------------------------------------------------
# 造假数据的小工具
# --------------------------------------------------------------------------
def fake_request():
    """openai 的异常对象要求带一个 request，这里造个假的（不会真发出去）。"""
    return httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def fake_response(status_code: int):
    return httpx.Response(status_code, request=fake_request())


def fake_completion(content: str):
    """造一个"长得像" openai 响应的假对象。

    mock 的第一原则：假对象不需要五脏俱全，只保留被测代码真正用到的属性即可。
    这里 chat() 只用到 .choices[0].message.content 两层，所以只造这两层。
    将来 chat() 改成读 response.usage.total_tokens，这里就得跟着补 ——
    这种"假对象一被打到就报错"的特性，反而顺便帮你盯住了代码对响应的依赖范围。
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_client(max_retries=0, reply="测试回复"):
    """造一个注入了假 client 的 DeepSeekClient。

    返回 (被测对象, create 这个 mock) —— 后者用来断言调用次数和入参。
    注意：全程不读 .env、不需要真实 API Key、不需要网络。
    """
    fake_sdk = MagicMock()
    create = fake_sdk.chat.completions.create
    create.return_value = fake_completion(reply)
    client = DeepSeekClient(client=fake_sdk, max_retries=max_retries)
    return client, create


class ClientTestCase(TestCase):
    """把被测代码里的 print 吞掉，让测试输出只剩下结果。
    想看重试日志的话，把这两个方法注释掉再跑就行。
    """

    def setUp(self):
        self._captured = io.StringIO()
        self._redirect = redirect_stdout(self._captured)
        self._redirect.__enter__()

    def tearDown(self):
        self._redirect.__exit__(None, None, None)


# --------------------------------------------------------------------------
# 正常路径：请求参数该长什么样
# --------------------------------------------------------------------------
class ChatRequestTests(ClientTestCase):
    def test_正常返回模型内容(self):
        client, create = make_client(reply="杭州是浙江省的省会。")

        result = client.chat(messages=[{"role": "user", "content": "用一句话介绍杭州"}])

        self.assertEqual(result, "杭州是浙江省的省会。")
        self.assertEqual(create.call_count, 1)

        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-flash")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "用一句话介绍杭州"}])
        self.assertFalse(kwargs["stream"])

    def test_没传response_format时不应带上这个键(self):
        """SDK 会把 None 原样序列化成 "response_format": null，
        部分 OpenAI 兼容端点收到它直接 400，所以这个键必须"根本不存在"。
        """
        client, create = make_client()

        client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertNotIn("response_format", create.call_args.kwargs)

    def test_显式传入的response_format要原样透传(self):
        client, create = make_client()

        client.chat(
            messages=[{"role": "user", "content": "输出 json"}],
            response_format={"type": "json_object"},
        )

        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_注入client后不再读env也不构造真OpenAI(self):
        """反向验证"测试不花钱"的前提真的成立：
        只要注入了 client，就不该碰 .env，也不该 new 出真 SDK 客户端。
        """
        with patch("src.api_client.load_dotenv") as mocked_load_dotenv, \
             patch("src.api_client.OpenAI") as mocked_openai:
            make_client()

        mocked_load_dotenv.assert_not_called()
        mocked_openai.assert_not_called()


# --------------------------------------------------------------------------
# 流式输出：拼接逻辑
# --------------------------------------------------------------------------
class StreamTests(ClientTestCase):
    def test_流式分片拼成完整字符串(self):
        client, create = make_client()
        contents = ["西", "湖", None, "好"]  # 首尾分片的 content 可能是 None
        create.return_value = iter(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )
            for text in contents
        )

        result = client.chat(messages=[{"role": "user", "content": "写诗"}], stream=True)

        self.assertEqual(result, "西湖好")
        self.assertTrue(create.call_args.kwargs["stream"])


# --------------------------------------------------------------------------
# 异常与重试：平时根本造不出来的分支，现在都能测
# --------------------------------------------------------------------------
class RetryTests(ClientTestCase):
    def test_超时一次后重试成功(self):
        client, create = make_client(max_retries=2)
        create.side_effect = [
            APITimeoutError(request=fake_request()),
            fake_completion("第二次成功了"),
        ]

        with patch("src.api_client.time.sleep") as sleeper:
            result = client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, "第二次成功了")
        self.assertEqual(create.call_count, 2)
        # 退避等待被 patch 掉了：既验证了等待秒数，又不让测试真睡 2 秒
        sleeper.assert_called_once_with(2)

    def test_重试次数用尽则返回兜底文案(self):
        client, create = make_client(max_retries=2)
        create.side_effect = APITimeoutError(request=fake_request())  # 每次调用都超时

        with patch("src.api_client.time.sleep"):
            result = client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, "抱歉，多次重试后仍然超时，请稍后再试。")
        self.assertEqual(create.call_count, 3)  # 首次调用 + 2 次重试

    def test_限流429会重试(self):
        client, create = make_client(max_retries=1)
        create.side_effect = [
            RateLimitError("rate limited", response=fake_response(429), body=None),
            fake_completion("限流后恢复"),
        ]

        with patch("src.api_client.time.sleep") as sleeper:
            result = client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, "限流后恢复")
        self.assertEqual(create.call_count, 2)
        sleeper.assert_called_once_with(2)

    def test_网络连接失败会重试(self):
        client, create = make_client(max_retries=1)
        create.side_effect = [
            APIConnectionError(request=fake_request()),
            fake_completion("网络恢复"),
        ]

        with patch("src.api_client.time.sleep"):
            result = client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, "网络恢复")
        self.assertEqual(create.call_count, 2)

    def test_401这类错误绝不重试(self):
        """这条最省钱：401/403 重试多少次都一样失败，重试只是在白烧配额。
        max_retries 特意设成 3，用来证明它一次都没用。
        """
        client, create = make_client(max_retries=3)
        create.side_effect = APIStatusError(
            "invalid api key", response=fake_response(401), body=None
        )

        with patch("src.api_client.time.sleep") as sleeper:
            result = client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, "API错误(401)，请检查API Key或账户余额。")
        self.assertEqual(create.call_count, 1)
        sleeper.assert_not_called()

    def test_未知异常兜底而不炸穿调用方(self):
        client, create = make_client()
        create.side_effect = ValueError("boom")

        result = client.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertIn("发生未知错误", result)
        self.assertIn("boom", result)


# --------------------------------------------------------------------------
# 结构化提取：往上一层，验证"校验—修复"闭环
# --------------------------------------------------------------------------
class ScriptedClient:
    """按剧本依次吐出预设回复的假客户端，同时记录每次收到的对话历史。

    extract_with_retry 只要求 ai_client 有一个 .chat() 方法，
    所以连 MagicMock 都不用，手写 8 行就够了 —— 这也是"面向接口"的好处：
    只要能注入，就能替换。
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, response_format=None):
        self.calls.append([dict(message) for message in messages])
        return self.replies.pop(0)


class StructuredOutputTests(ClientTestCase):
    def test_一次就拿到合法json(self):
        ai = ScriptedClient(['{"name": "张三", "years": 5, "skills": ["Java"]}'])

        result = extract_with_retry(
            ai,
            [{"role": "user", "content": "张三，5 年 Java"}],
            ResumeInfo,
        )

        self.assertEqual(result.name, "张三")
        self.assertEqual(result.years, 5)
        self.assertEqual(result.skills, ["Java"])
        self.assertEqual(len(ai.calls), 1)  # 没有多余请求 = 没有多余花费

    def test_非法json会带上错误报告要求模型重写(self):
        ai = ScriptedClient([
            "这不是 JSON，我是模型我任性",
            '{"name": "李四", "years": 3, "skills": ["Python"]}',
        ])

        result = extract_with_retry(
            ai,
            [{"role": "user", "content": "李四，3 年 Python"}],
            ResumeInfo,
            max_retries=1,
        )

        self.assertEqual(result.name, "李四")
        self.assertEqual(len(ai.calls), 2)
        # 第二次请求的历史被追加了 1 轮：(模型的原回复) + (错误报告)
        self.assertEqual(
            [message["role"] for message in ai.calls[1]],
            ["user", "assistant", "user"],
        )
        self.assertIn("请严格按 JSON 格式重新输出", ai.calls[1][-1]["content"])

    def test_业务校验不过也会触发修复(self):
        """JSON 语法合法，但 years=999 超出 Field(ge=0, le=50)：
        该走的是同一套修复闭环，而不是让脏数据流到调用方手里。
        """
        ai = ScriptedClient([
            '{"name": "王五", "years": 999, "skills": ["Go"]}',
            '{"name": "王五", "years": 9, "skills": ["Go"]}',
        ])

        result = extract_with_retry(
            ai,
            [{"role": "user", "content": "王五，9 年 Go"}],
            ResumeInfo,
            max_retries=1,
        )

        self.assertEqual(result.years, 9)
        self.assertEqual(len(ai.calls), 2)

    def test_重试用尽要把异常抛给调用方(self):
        """降级策略是调用方的决定，函数不该自己吞掉错误返回半成品。"""
        ai = ScriptedClient(["垃圾一", "垃圾二"])

        with self.assertRaises(ValidationError):
            extract_with_retry(
                ai,
                [{"role": "user", "content": "谁"}],
                ResumeInfo,
                max_retries=1,
            )

        self.assertEqual(len(ai.calls), 2)

    def test_不改动调用方传进来的对话历史(self):
        """函数内部做了 list() 拷贝，所以原历史不该被污染。"""
        history = [{"role": "user", "content": "张三，5 年 Java"}]
        ai = ScriptedClient([
            "坏数据",
            '{"name": "张三", "years": 5, "skills": ["Java"]}',
        ])

        extract_with_retry(ai, history, ResumeInfo, max_retries=1)

        self.assertEqual(history, [{"role": "user", "content": "张三，5 年 Java"}])


if __name__ == "__main__":
    unittest_main(verbosity=2)
