import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

from p0.paths import DEFAULT_PATHS

SHOWDOWN_DIR = DEFAULT_PATHS.showdown_root


class ShowdownServer:
    def __init__(self, port: int = 8000):
        self.port = port
        self.proc = None
        self._old_sigterm_handler = None
        self._old_sigint_handler = None
        self._shutdown_requested = False

    def start(self):
        print(f"[Showdown Server] Building pokemon-showdown in {SHOWDOWN_DIR}...")
        try:
            subprocess.run(
                ["node", "build"],
                cwd=SHOWDOWN_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            print("[Showdown Server] Pokemon-showdown built successfully.")
        except Exception as e:
            print(f"[Showdown Server] Failed to build pokemon-showdown: {e}", file=sys.stderr)
            raise e

        # Clean up other processes occupying the port
        try:
            subprocess.run(
                ["fuser", "-k", f"{self.port}/tcp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        print(f"[Showdown Server] Starting Showdown server on port {self.port}...")
        self.proc = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", "--skip-build", str(self.port)],
            cwd=SHOWDOWN_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait and check for the port to be listening
        start_time = time.time()
        timeout = 30.0
        success = False
        while (time.time() - start_time) < timeout:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    success = True
                    break
            except OSError:
                pass
            time.sleep(0.5)

        if not success:
            self.stop()
            raise RuntimeError(f"Showdown server failed to start on port {self.port}")
        print(f"[Showdown Server] Showdown server is ready on port {self.port}.")

        # Set up signal handlers for clean shutdown
        def handle_signal(signum, frame):
            if self._shutdown_requested:
                print(
                    "[Showdown Server] Second shutdown signal received, forcing immediate exit...",
                    file=sys.stderr,
                )
                os._exit(1)
            print(
                f"[Showdown Server] Signal {signum} received, terminating Showdown server...",
                file=sys.stderr,
            )
            self._shutdown_requested = True
            self.stop()
            sys.exit(1)

        self._old_sigterm_handler = signal.signal(signal.SIGTERM, handle_signal)
        self._old_sigint_handler = signal.signal(signal.SIGINT, handle_signal)

    def stop(self):
        if self.proc is not None:
            print("[Showdown Server] Stopping Showdown server...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print(
                    "[Showdown Server] Showdown server did not terminate in 5 seconds, killing process...",
                    file=sys.stderr,
                )
                self.proc.kill()
            self.proc = None
            print("[Showdown Server] Showdown server stopped.")

        # Restore original signal handlers
        if self._old_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm_handler)
            self._old_sigterm_handler = None
        if self._old_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._old_sigint_handler)
            self._old_sigint_handler = None


@contextlib.contextmanager
def spawned_showdown(port: int = 8000):
    server = ShowdownServer(port)
    server.start()
    try:
        yield server
    finally:
        server.stop()
