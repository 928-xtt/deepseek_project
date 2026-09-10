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


# 多轮对话入口
# def start_chat():
#
#     print("AI 助手已启动（输入 'quit' 退出）...")
#
#     # 1. 初始化客户端（这就像 Java 里的注入 Service）
#     ai_client = DeepSeekClient(max_retries=2, timeout=30.0)
#
#     # 2. 初始化对话历史（核心！大模型没有记忆，全靠你把历史传给它）
#     # system 角色用来设定 AI 的人设
#     history_messages = [
#         {"role": "system", "content": "你是一个友善的编程助手。"}
#     ]
#
#     while True:
#         # 3. 接收用户输入
#         user_input = input("\n我: ")
#
#         if user_input.strip().lower() in ['quit', 'exit', 'q']:
#             print("再见！")
#             break
#
#         # 4. 将用户输入放入历史记录
#         history_messages.append({"role": "user", "content": user_input})
#
#         try:
#             # 5. 调用大模型，把完整历史传过去
#             print("AI: ", end="", flush=True)
#             ai_reply = ai_client.chat(history_messages, stream=True, temperature=0.7)
#
#             # stream=True 时，_handle_stream_response 已经边打印边返回了完整字符串
#             # stream=False 时，上面这行需要改成：
#             # ai_reply = ai_client.chat(history_messages, stream=False)
#             # print(ai_reply)
#
#             # 6. 将 AI 的回复也放入历史记录（这样下一轮对话时，AI 才能记住刚才说了什么）
#             history_messages.append({"role": "assistant", "content": ai_reply})
#
#         except Exception as e:
#             print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
