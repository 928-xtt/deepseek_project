# 程序的唯一入口

from models.ResumeInfo import ResumeInfo
from pydantic import ValidationError
from src.api_client import DeepSeekClient
import json
from src.jisoModeAndPydantic import extract_with_retry


# 【依赖注入改造 1/4】模块顶部原来有一行
#     ai_client = DeepSeekClient(max_retries=2, timeout=20.0)
# 它有两个问题，所以删掉：
# 1. import 副作用：只要有人 import main，就会读 .env、建连接池；
#    以后想写单元测试 import 里面的函数，会先把真客户端建起来。
# 2. 隐式全局依赖：下面每个函数用到的 ai_client 从哪来，看函数签名看不出来；
#    而 structured_output 里又自己 new 了一个，同一个文件两个实例打架。
# 现在改成：客户端只在入口处 new 一次（组合根），然后显式当参数传下去。
def build_client() -> DeepSeekClient:
    """唯一构造客户端的地方（组合根），模型名/超时等配置只在这里改一次。"""
    return DeepSeekClient(max_retries=2, timeout=20.0)


def main(client: DeepSeekClient):
    print("初始化大模型客户端...")

    # 测试非流式
    print("\n--- 非流式调用 ---")
    # 【依赖注入改造 2/4】这里原来用全局 ai_client 或自己 new 一个，
    # 现在统一用传进来的 client：谁创建、谁传参，在调用点上就能看清。
    result = client.chat(
        messages=[{"role": "user", "content": "用一句话介绍杭州"}],
        stream=False,
        temperature=0.5
    )
    print(result)

    # 测试流式
    print("\n--- 流式调用 ---")
    # client.chat(messages=[{"role": "user", "content": "写一首关于西湖的诗"}], stream=True)



# 结构化输出
def structured_output(client: DeepSeekClient):
    print("初始化大模型客户端...")
    # 【依赖注入改造 3/4】删掉了这里原来的局部
    #     ai_client = DeepSeekClient(max_retries=2, timeout=20.0)
    # 同一个程序里维护多个配置可能不一致的客户端，是重复来源也是坑。

    # DeepSeek JSON Mode
    result = client.chat(
        messages= [
            {"role": "system", "content": "你是一个信息助手，始终以 json 格式输出。"},
            {"role": "user", "content": "用一句话介绍杭州"}
        ],
        stream=False,
        temperature=0.5,
        response_format={"type": "json_object"}
    )
    # print("-----------------------------------------------------------------\n"+repr(result)+"\n---------------------------------------------------")
    data = json.loads(result)
    print(data)


# 多轮对话
def start_chat(client: DeepSeekClient):

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

    sys_prompt = "你是一名专业的心理医生，同时也是一名专业的哲学大师 "

    # history_messages = [
    #     {"role": "system", "content": "你是一个助手。"}
    # ]
    history_messages = [
        {"role": "system", "content": sys_prompt},
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
        reply  = client.chat(
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

def model_json(client: DeepSeekClient):
    history_messages = [
        {"role": "system", "content": "从简历提取信息，以 json 格式输出，字段: name, years, skills"},
        {"role": "user", "content": "刘七，资深工程师，工作经验丰富，技术栈很广"}  # 👈 故意模糊
    ]

    try:
        result = extract_with_retry(
            client,
            history_messages,
            ResumeInfo,
            response_format={"type": "json_object"},  # 开 JSON Mode，让模型直接吐 JSON
        )
        print(f"输出结果为{result}")
    except ValidationError as e:
        # 【依赖注入改造 4/4】原来是裸 `except:`，它会把 TypeError、KeyError 这类
        # 真正的 bug 一起吞掉，再统一打印成"格式化 json 错误"，排查方向直接被带偏
        # ——这次 response_format 的 TypeError 就是被它掩盖成"模型不听话"的。
        # 只捕获预期内的校验失败，并把真实错误打出来。
        print(f"模型返回内容不是合法 JSON 或不符合 schema：{e}")


if __name__ == "__main__":
    # 【依赖注入改造·收口】客户端只在入口构造一次，再传给要跑的那个函数。
    # 想换模型/超时只改 build_client() 一处；想换实验只改下面这一行。
    client = build_client()
    # main(client)
    # start_chat(client)
    # structured_output(client)
    # model_json(client)
    summarize(client)
