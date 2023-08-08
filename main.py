from src.launcher import Launcher


if __name__ == "__main__":
  launcher = Launcher(checkpoint="t5-small")
  launcher.run()
