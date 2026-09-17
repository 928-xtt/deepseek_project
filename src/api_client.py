# 核心：大模型调用与重试封装
import os
import time
from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
import tiktoken


class DeepSeekClient:
    """
    大模型调用封装类 (类似 Java 里的 DeepSeekService)
    """

    # 【依赖注入改造 1/3】构造注入：这个类只负责"怎么调"，"调到哪、用哪个模型"
    # 属于它的配置，应该由构造函数传进来，而不是写死在 chat() 里。
    # - model / base_url 提成参数：换模型、换兼容端点（比如本地 vLLM）不用改业务方法，
    #   同一个进程里也能同时持有两个不同模型的客户端。
    # - client 参数：允许外部注入底层 SDK 客户端。测试时传入一个假客户端就能
    #   断言重试逻辑，既不用联网也不需要真的 key。
    def __init__(
        self,
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_retries=0,
        timeout=30.0,
        client=None,
    ):
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        if client is not None:
            # 外部注入时不再读 .env、不再新建连接池：配置权完全交给调用方
            self.client = client
        else:
            load_dotenv()
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url,
                timeout=timeout,
            )

    # def chat(self, prompt, stream=False, temperature=1.0):
    def chat(self, messages, stream=False, temperature=1.0, response_format=None):
        """统一的大模型调用入口

        【依赖注入改造 2/3】为什么这里必须显式声明 response_format 并透传：
        它是"调用方决定要不要 JSON Mode"的传输层选项，属于入参而不是实现细节。
        之前 main.structured_output() 传了它、这里没声明，Python 在进入函数体
        之前就抛 TypeError: unexpected keyword argument，重试逻辑根本没机会执行。
        """
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # 【模型名不再写死】用构造时传入的 self.model，换模型只改一处配置
                params = {
                    "model": self.model,
                    "messages": messages,
                    "stream": stream,
                    "temperature": temperature,
                }
                # 只在显式传入时才带上这个键：SDK 会把 None 原样序列化进请求体，
                # 部分 OpenAI 兼容端点收到 "response_format": null 会直接报 400
                if response_format is not None:
                    params["response_format"] = response_format

                response = self.client.chat.completions.create(**params)
                print(
                    f"----------------------------本轮输入 token: {response.usage.prompt_tokens}----------------------------"
                )
                if stream:
                    return self._handle_stream_response(response)
                else:
                    return response.choices[0].message.content

                    # ---------------- 超时处理 ----------------
            except APITimeoutError:
                retry_count += 1
                wait_time = 2**retry_count
                print(f"⏰ 请求超时，第 {retry_count} 次重试，等待 {wait_time} 秒...")
                if retry_count > self.max_retries:
                    return "抱歉，多次重试后仍然超时，请稍后再试。"
                time.sleep(wait_time)
                # ---------------- 429 限流处理 ----------------
            except RateLimitError:
                retry_count += 1
                wait_time = 2**retry_count
                print(
                    f"🚦 触发限流(429)，第 {retry_count} 次重试，等待 {wait_time} 秒..."
                )
                if retry_count > self.max_retries:
                    return "调用太频繁了，请稍后再试。"
                time.sleep(wait_time)
                # ---------------- 网络连接问题 ----------------
            except APIConnectionError:
                retry_count += 1
                wait_time = 2**retry_count
                print(
                    f"🔌 网络连接失败，第 {retry_count} 次重试，等待 {wait_time} 秒..."
                )
                if retry_count > self.max_retries:
                    return "网络连接异常，请检查网络后重试。"
                time.sleep(wait_time)
                # ---------------- 其他 API 错误（如 401 Key错误） ----------------
            except APIStatusError as e:
                print(f"❌ API返回错误，状态码: {e.status_code}")
                print(f"错误信息: {e.message}")
                # 401/403 这种错误重试也没用，直接返回
                return f"API错误({e.status_code})，请检查API Key或账户余额。"

            # ---------------- 兜底：未知异常 ----------------
            except Exception as e:
                print(f"未知异常: {type(e).__name__}: {e}")
                return f"发生未知错误: {e}"

    def _handle_stream_response(self, response):
        """处理流式输出的私有方法"""
        full_content = ""
        for chunk in response:
            # print(chunk)
            content = chunk.choices[0].delta.content
            # 注意：流式输出时，第一个或最后一个 chunk 的 content 可能是 None
            if content:
                print(
                    content, end="", flush=True
                )  # end的意义是不让流式输出的打印结果换行      flush的作用是直接打印响应结果，避免python接收响应结果后缓存
                full_content += content
        return full_content

    # N轮对话裁剪

def sliding_window(messages: list[dict], max_turns: int = 10) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    return system + history[-(max_turns * 2) :]  # 一轮 = user + assistant


# 在调用llm之前,计算本次对话输入消耗的token(包含了中文和英文两种语言)
enc = tiktoken.get_encoding("cl100k_base")  # DeepSeek 与 OpenAI tokenizer 兼容

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


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
    print(resp)
    return resp
