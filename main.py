# 程序的唯一入口
from src.api_client import DeepSeekClient

ai_client = DeepSeekClient(max_retries=2, timeout=20.0)

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

def start_chat():

    ######## 自己写的 System Prompt 包含四要素：角色 + 约束 + 格式 + 兜底 ########

    sys_prompt1 = """
    
        # 角色
        你是一位有 10 年经验的资深 Java 后端工程师，同时精通 Python。
        你正在指导一位有 Java 基础、正在学 Python 和 LLM 应用开发的学习者。

        # 沟通风格
        - 先给结论，再给解释
        - 涉及 Python 概念时，主动用 Java 类比帮助理解
        - 代码示例保持简洁，不超过 30 行

        # 约束
        - 只回答与编程、LLM 开发相关的问题,其他无关问题直接拒绝回答
        - 不确定的技术细节明确说"我不确定"，不要编造 API 或类名
        - 不推荐已弃用的库或写法

        # 输出格式
        1. 结论（1-2 句话）
        2. 代码示例（带注释）
        3. Java 类比（如适用）
        4. 常见坑（1 条即可）

    
    """

    sys_prompt2 = """
        # 角色
            你是"XX 电商"的售后客服专员，服务对象是普通消费者。
            
        # 任务范围
            可以处理：订单查询、退换货流程说明、物流问题指引。
            不能处理：改密码、退款审批、价格补偿、投诉仲裁——这些要转人工。
            
        # 沟通规则
            - 语气友好但简洁，每次回复不超过 150 字
            - 一次只问一个澄清问题
            - 不索取密码、身份证号等敏感信息
            
        # 信息来源
            - 产品和售后政策只能依据知识库内容回答
            - 知识库没有的信息，回复："抱歉，这个问题我需要为您转接人工客服"
            
        # 输出格式
            1. 对用户问题的简要确认（1 句）
            2. 解决方案（分步骤，最多 3 步）
            3. 是否需要转人工（是/否）
    """

    # history_messages = [
    #     {"role": "system", "content": "你是一个助手。"}
    # ]
    history_messages = [
        {"role": "system", "content": sys_prompt1},
    ]

    print("开始对话（输入 'quit' 退出）")

    while True:
        # 用户输入
        user_input = input("\n你:").strip()
        if user_input.lower() in ("quit", "exit","q"):
            print("再见！")
            break
        if not user_input:
            continue
        # 拼接用户输入
        history_messages.append({"role": "user", "content": user_input})
        #调用大模型
        reply  = ai_client.chat(
            history_messages,
            stream=False,
            temperature=0.7,
        )
        #拼接大模型回复内容
        history_messages.append({"role": "assistant", "content": reply})
        print(f"AI: {reply}")

        # print(f"\n[DEBUG] 当前 history 长度: {len(history_messages)}")
        # for m in history_messages:
        #     print(f"  - {m['role']}: {m['content'][:40]}...")


if __name__ == "__main__":
    # main()
    start_chat()
