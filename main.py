import os
import asyncio

os.environ['WECHATY_PUPPET'] = 'wechaty-puppet-service'
os.environ['WECHATY_PUPPET_SERVICE_TOKEN'] = 'puppet_padlocal_xxxx'

from wechaty import (
    Contact,
    Message,
    Wechaty,
    ScanStatus,
)
from world import World
world = World()

async def on_message(msg: Message):
    """
    Message Handler for the Bot
    """
    if msg.talker().name == '古德猫宁':
        if msg.text() == '开始':
            # 向用户输出故事开头
            await msg.say(world.start())
            return

        response = world.receive(msg.text())
        await msg.say(response)

async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    """
    Scan Handler for the Bot
    """
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    """
    Login Handler for the Bot
    """
    print(user)
    # TODO: To be written

async def main():
    """
    Async Main Entry
    """
    #
    # Make sure we have set WECHATY_PUPPET_SERVICE_TOKEN in the environment variables.
    # Learn more about services (and TOKEN) from https://wechaty.js.org/docs/puppet-services/
    #
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()

    bot.on('scan',      on_scan)
    bot.on('login',     on_login)
    bot.on('message',   on_message)

    await bot.start()

    print('[Python Wechaty] Ding Dong Bot started.')


asyncio.run(main())