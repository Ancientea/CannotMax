def main():
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise ImportError(
            "Gradio可视化依赖未安装。请执行: uv sync --extra gradio"
        )
    from gradio_ml.__main__ import main as gradio_main
    gradio_main()


if __name__ == "__main__":
    main()
