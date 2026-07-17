"""AI 智能助手 — 启动入口

使用方法：
    python main.py                  # 启动 Gradio 界面
    python main.py --port 7861      # 指定端口
"""
import argparse

from app.gradio_ui import AgentUI


def main():
    parser = argparse.ArgumentParser(description="AI 智能助手")
    parser.add_argument("--port", type=int, default=7860, help="服务端口号")
    parser.add_argument("--share", action="store_true", help="是否生成公网链接")
    args = parser.parse_args()

    print("🚀 正在启动 AI 智能助手...")
    print(f"📌 本地地址: http://127.0.0.1:{args.port}")
    if args.share:
        print("🌐 将生成公网分享链接（请注意数据安全）")

    ui = AgentUI()
    ui.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
