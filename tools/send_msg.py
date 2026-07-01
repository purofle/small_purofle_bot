from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

__tools_name__ = "send_message"
__tools_description__ = "Send a message to a Telegram chat."


class SendMessageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat_id: int = Field(..., description="Target chat id")
    text: str = Field(..., min_length=1, max_length=4096, description="Message text")
    reply_to_message_id: Optional[int] = Field(None, description="Message id to reply to")


__tools_parameters__ = SendMessageArgs.model_json_schema()


async def send_message(ctx, **kwargs):
    args = SendMessageArgs.model_validate(kwargs)
    if args.reply_to_message_id is not None:
        msg = await ctx.bot.send_message(
            args.chat_id,
            args.text,
            reply_to_message_id=args.reply_to_message_id
        )
    else:
        msg = await ctx.bot.send_message(
            args.chat_id,
            args.text
        )

    return {"message_id": msg.message_id}
