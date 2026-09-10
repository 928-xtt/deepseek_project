# 核心：大模型调用与重试封装
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError, APITimeoutError


class DeepSeekClient:
    """
        大模型调用封装类 (类似 Java 里的 DeepSeekService)
    """
    def __init__(self, max_retries=3, timeout=30.0):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            timeout=timeout
        )
        self.max_retries = max_retries

    def chat(self, prompt, stream=False, temperature=1.0):
        """统一的大模型调用入口"""
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": prompt}],
                    stream=stream,
                    temperature=temperature
                )
                if stream:
                    return self._handle_stream_response(response)
                else:
                    return response.choices[0].message.content

                    # ---------------- 超时处理 ----------------
            except APITimeoutError:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"⏰ 请求超时，第 {retry_count} 次重试，等待 {wait_time} 秒...")
                if retry_count > self.max_retries:
                    return "抱歉，多次重试后仍然超时，请稍后再试。"
                time.sleep(wait_time)
                    # ---------------- 429 限流处理 ----------------
            except RateLimitError:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"🚦 触发限流(429)，第 {retry_count} 次重试，等待 {wait_time} 秒...")
                if retry_count > self.max_retries:
                    return "调用太频繁了，请稍后再试。"
                time.sleep(wait_time)
                # ---------------- 网络连接问题 ----------------
            except APIConnectionError:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"🔌 网络连接失败，第 {retry_count} 次重试，等待 {wait_time} 秒...")
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
                print(content, end="", flush=True)  # end的意义是不让流式输出的打印结果换行      flush的作用是直接打印响应结果，避免python接收响应结果后缓存
                full_content += content
        return full_content