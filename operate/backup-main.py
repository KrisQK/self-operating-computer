"""
Self-Operating Computer
"""

# 导入所需的库和模块
import os
import time
import base64
import json
import math
import re
import subprocess
import pyautogui
import argparse
import platform
import Xlib.display  # Xlib用于Linux系统的屏幕截图

# 以下是导入的一些特定功能库
from prompt_toolkit import prompt  # 创建交互式命令行界面
from prompt_toolkit.shortcuts import message_dialog  # 显示消息对话框
from prompt_toolkit.styles import Style as PromptStyle  # 定义界面样式
from dotenv import load_dotenv  # 从.env文件中加载环境变量
from PIL import Image, ImageDraw, ImageFont, ImageGrab  # 图像处理库
import matplotlib.font_manager as fm  # 用于查找系统字体
from openai import OpenAI  # OpenAI API交互库

# 加载环境变量
load_dotenv()

# 是否开启调试模式
DEBUG = False

# 创建OpenAI客户端
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)  # 设置API基础URL，默认为client.base_url

# 以下是一些命令和交互信息的模板，用于指导自操作计算机完成任务
# 通过命令行与用户进行交互以获取任务指令，例如键盘输入、鼠标点击、搜索程序等
VISION_PROMPT = """
...（这里是具体的指导模板，告诉自操作计算机如何执行任务，如何回应用户等）
"""

# 在自动执行任务完成后，总结结果并呈现给用户
SUMMARY_PROMPT = """
...（这里是总结任务执行结果的模板）
"""

# 定义一个自定义异常类，用于处理未识别的模型异常
class ModelNotRecognizedException(Exception):
    """Exception raised for unrecognized models."""

    def __init__(self, model, message="Model not recognized"):
        self.model = model
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} : {self.model} "


# 定义样式
style = PromptStyle.from_dict(
    {
        # 样式设置，如颜色、背景等
    }
)

# 定义ANSI转义码，用于命令行文本样式设置
ANSI_GREEN = "\033[32m"
# 其他ANSI码设置...

# 主程序入口
def main(model):
    """
    主函数，实现自操作计算机的主要功能
    """

    # 创建交互式命令行界面
    message_dialog(
        title="Self-Operating Computer",
        text="Ask a computer to do anything.",
        style=style,
    ).run()

    # 检查操作系统类型
    print("SYSTEM", platform.system())

    # 根据不同的操作系统清除命令行界面
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    # 提示用户输入任务指令
    print(f"{ANSI_GREEN}[Self-Operating Computer]\n{ANSI_RESET}{USER_QUESTION}")
    print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")

    # 从命令行获取用户输入的任务指令
    objective = prompt(
        style=style,
    )

    # 创建用户和助手（自操作计算机）的消息列表
    assistant_message = {"role": "assistant", "content": USER_QUESTION}
    user_message = {
        "role": "user",
        "content": f"Objective: {objective}",
    }
    messages = [assistant_message, user_message]

    loop_count = 0

    while True:
        # 如果启用了调试模式，打印消息
        if DEBUG:
            print("[loop] messages before next action:\n\n\n", messages[1:])
        
        # 尝试从OpenAI获取下一步操作
        try:
            response = get_next_action(model, messages, objective)
            action = parse_oai_response(response)
            action_type = action.get("type")
            action_detail = action.get("data")

        # 捕获未识别模型的异常
        except ModelNotRecognizedException as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break
        except Exception as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break

        # ... 省略部分代码

        loop_count += 1
        if loop_count > 10:
            break

# 更多代码...

# 以下是格式化任务总结提示的函数
def format_summary_prompt(objective):
    """
    格式化任务总结提示
    """
    prompt = SUMMARY_PROMPT.format(objective=objective)
    return prompt


# 以下是格式化任务指导提示的函数
def format_vision_prompt(objective, previous_action):
    """
    格式化任务指导提示
    """
    if previous_action:
        previous_action = f"Here was the previous action you took: {previous_action}"
    else:
        previous_action = ""
    prompt = VISION_PROMPT.format(objective=objective, previous_action=previous_action)
    return prompt


# 获取下一步操作的函数
def get_next_action(model, messages, objective):
    if model == "gpt-4-vision-preview":
        content = get_next_action_from_openai(messages, objective)
        return content
    elif model == "agent-1":
        return "coming soon"

    raise ModelNotRecognizedException(model)


# 获取最后一条助手消息的函数
def get_last_assistant_message(messages):
    """
    获取最后一条助手消息
    """
    for index in reversed(range(len(messages))):
        if messages[index]["role"] == "assistant":
            if index == 0:  # 检查助手消息是否是数组中的第一个消息
                return None
            else:
                return messages[index]
    return None  # 如果没有找到助手消息，则返回None


# 从OpenAI获取下一步操作的函数
def get_next_action_from_openai(messages, objective):
    """
    从OpenAI获取自操作计算机的下一步操作
    """
    # ... 省略部分代码

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=pseudo_messages,
        presence_penalty=1,
        frequency_penalty=1,
        temperature=0.7,
        max_tokens=300,
    )

    # ... 省略部分代码

    content = response.choices[0].message.content
    return content


# 解析OpenAI响应的函数
def parse_oai_response(response):
    # ... 解析OpenAI的响应并返回相应类型和数据 ...


# 总结任务执行结果的函数
def summarize(messages, objective):
    # ... 总结任务执行的结果并返回 ...


# 鼠标点击操作的函数
def mouse_click(click_detail):
    # ... 实现鼠标点击操作 ...


# 键盘输入操作的函数
def keyboard_type(text):
    # ... 实现键盘输入操作 ...


# 搜索程序操作的函数
def search(text):
    # ... 实现搜索程序操作 ...


# 获取带有光标的屏幕截图的函数
def capture_screen_with_cursor(file_path=os.path.join("screenshots", "screenshot.png")):
    # ... 获取带有光标的屏幕截图 ...


# 从字符串中提取JSON数据的函数
def extract_json_from_string(s):
    # ... 从字符串中提取JSON数据 ...


# 将百分比转换为小数的函数
def convert_percent_to_decimal(percent_str):
    # ... 将百分比转换为小数 ...


# 主程序入口
def main_entry():
    # ... 解析命令行参数并调用主函数 ...


if __name__ == "__main__":
    main_entry()
