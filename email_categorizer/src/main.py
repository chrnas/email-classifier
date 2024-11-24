from cli import Cli
from client import Client


if __name__ == '__main__':
    client = Client()
    cli = Cli(client)
    cli.run()
