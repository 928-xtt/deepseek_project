# 程序的唯一入口
from src.api_client import DeepSeekClient


def main():
    print("初始化大模型客户端...")
    ai_client = DeepSeekClient(max_retries=2, timeout=20.0)

    # 测试非流式
    print("\n--- 非流式调用 ---")
    result = ai_client.chat("用一句话介绍杭州", stream=False, temperature=0.5)
    print(result)

    # 测试流式
    print("\n--- 流式调用 ---")
    ai_client.chat("写一首关于西湖的诗", stream=True)


if __name__ == "__main__":
    main()
