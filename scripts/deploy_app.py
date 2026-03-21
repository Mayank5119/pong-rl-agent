"""
Deploy: Run Gradio web interface
"""

from components.app import create_gradio_interface


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌐 PONG RL AGENT - GRADIO DEPLOYMENT")
    print("=" * 70)
    print()
    print("Make sure Docker server is running:")
    print("  docker run -p 8000:8000 pong-server")
    print()
    print("Launching Gradio interface...")
    print()

    demo = create_gradio_interface()
    demo.launch(share=True)

