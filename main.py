import sys
from os import getenv

import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from openai import AsyncOpenAI

from AIBot import AIBot

TOKEN = getenv("BOT_TOKEN")
SUPERUSER_ID = int(getenv("SUPERUSER_ID", "-1002978097105"))

ai_client = AsyncOpenAI(
    api_key=getenv("OPENAI_API_KEY"),
    base_url=getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
)

ai_bot: AIBot

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await ai_bot.chat(message, reset_history=True)


@dp.message()
async def message_handler(message: Message) -> None:
    text = (message.text or "").strip()
    if text.startswith("/"):
        return
    await ai_bot.chat(message)


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    global ai_bot
    ai_bot = AIBot(
        telegram_bot=bot,
        client=ai_client,
        model_name=getenv("OPENAI_MODEL_NAME", "gpt-5.2-chat"),
        superuser_id=SUPERUSER_ID,
    )
    # And the run events dispatching
    await bot.delete_webhook(True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.DEBUG)
    asyncio.run(main())
