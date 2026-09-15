from pydantic import ValidationError, BaseModel
from models.ResumeInfo import ResumeInfo


# 【依赖注入改造 3/3】这里原本有一行 `ai_client = DeepSeekClient()`，已删除。原因：
# 1. import 副作用：任何人 import 本模块都会顺带读 .env、建 HTTP 连接池，
#    配置缺失时连 import 都会失败 —— 模块"能否导入"不该取决于运行环境。
# 2. 它和 main.py 模块顶部的 ai_client、以及 extract_with_retry 收到的参数，
#    是三个互不相干的实例，"现在到底在用哪个客户端"从调用点看不出来。
# 3. 它本身就是死代码：extract_with_retry 的签名早就在收 ai_client 参数了，
#    函数体从头到尾没碰过这个模块级实例。
# 现在客户端由调用方构造一次、一路传下去（入口见 main.py 的 build_client）。


def extract_with_retry(
    ai_client,
    history_messages,
    schema: type[BaseModel],
    max_retries: int = 2,
    response_format=None,
):
    """带校验-重试闭环的结构化提取

    ai_client 由调用方注入（不在这里 new）；response_format 原样透传给客户端，
    想开启 DeepSeek JSON Mode 就传 {"type": "json_object"}。
    注意：DeepSeek 要求 messages 里出现 "json" 字样，否则 JSON Mode 会返回 400。
    """
    messages = list(history_messages)  # 拷贝，别污染原对话历史

    for attempt in range(max_retries + 1):
        reply = ai_client.chat(messages=messages, response_format=response_format)
        # print(f"当前错误内容为{reply}")
        try:
            print("------------------------------------"+f"尝试第{attempt+1}次"+"--------------------------\n"+f"模型返回内容{reply}")
            return schema.model_validate_json(reply)  # ✅ 成功：返回 Pydantic 对象
        except ValidationError as e:
            if attempt == max_retries:
                raise  # 重试用尽，向上抛，让调用方决定降级策略

            # 👇 精髓：把"错误报告"作为 user 消息追加，让模型自己修
            error_report = f"你上一次的输出不是合法 JSON 或不符合要求，错误如下：\n{e.errors()}\n请严格按 JSON 格式重新输出，不要有任何多余文字。"
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": error_report})

# 用法
# result = extract_with_retry(
#     ai_client,
#     [{"role": "system", "content": "从简历提取信息，以 json 格式输出，字段：name, years, skills"},
#      {"role": "user", "content": "张三，5 年 Java，会 Spring Boot 和 MySQL"}],
#     ResumeInfo
# )
# print(result)
# print(result.name, result.years)  # 类型安全的字段访问
