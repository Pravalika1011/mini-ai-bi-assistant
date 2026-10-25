"""
Runner script to launch both the FastAPI backend and Streamlit frontend concurrently.
Usage:
    python run.py
"""

import os
import sys
import subprocess
import time
import signal

def main():
    print("Starting RSG Mini AI Business Intelligence Assistant...")

    # Determine python executable
    python_exe = sys.executable

    backend_cmd = [
        python_exe, "-m", "uvicorn", "app.api:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]

    frontend_cmd = [
        python_exe, "-m", "streamlit", "run",
        os.path.join("frontend", "streamlit_app.py"),
        "--server.port", "8501"
    ]

    print(f"Launching Backend API on http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(backend_cmd)

    # Allow backend 1 second to start listening
    time.sleep(1.5)

    print(f"Launching Streamlit Frontend on http://localhost:8501...")
    frontend_proc = subprocess.Popen(frontend_cmd)

    print("Both services are running. Press Ctrl+C to terminate.")

    try:
        while True:
            # Check if either process terminated unexpectedly
            b_poll = backend_proc.poll()
            f_poll = frontend_proc.poll()
            if b_poll is not None:
                print(f"Backend process terminated with code {b_poll}")
                break
            if f_poll is not None:
                print(f"Frontend process terminated with code {f_poll}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in [backend_proc, frontend_proc]:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Services stopped.")

if __name__ == "__main__":
    main()
