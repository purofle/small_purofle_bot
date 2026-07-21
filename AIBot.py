from datetime import datetime

import asyncio
import base64
import io
import json
import logging
import mimetypes
from typing import Any

from aiogram import Bot
from aiogram.types import Message
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from tool_registry import ToolContext, ToolExecutor, load_tools

from llm_core.prompt import Prompt, PromptContext
from context_compression import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    DEFAULT_RESPONSE_TOKEN_RESERVE,
    compress_history,
    estimate_message_tokens,
    estimate_tokens,
)


def _is_private_chat(message: Message) -> bool:
    return str(message.chat.type) == "private"


def _message_text(message: Message) -> str:
    return (message.text or message.caption or "").strip()


class BotInputMessage(BaseModel):
    chat_id: int
    text: str
    date: datetime
    from_user_full_name: str
    message_id: int
    from_user_id: int
    media_notes: list[str] = Field(default_factory=list)


class GroupBatchItem(BaseModel):
    message_id: int
    text: str
    date: datetime
    from_user_full_name: str
    from_user_id: int
    mentions_bot: bool
    replies_to_bot: bool


class GroupBatchInput(BaseModel):
    chat_id: int
    batch_size: int
    priority_message_ids: list[int]
    messages: list[GroupBatchItem]
    media_notes: list[str] = Field(default_factory=list)


class AIBot:
    def __init__(
        self,
        telegram_bot: Bot,
        client: AsyncOpenAI,
        model_name: str,
        superuser_id: int,
        context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
        response_token_reserve: int = DEFAULT_RESPONSE_TOKEN_RESERVE,
    ) -> None:
        self.client: AsyncOpenAI = client
        self.model_name = model_name
        self.telegram_bot = telegram_bot
        self.superuser_id = superuser_id
        self.tools = load_tools("tools/")
        self.chat_histories: dict[int, list[dict[str, Any]]] = {}
        self.chat_locks: dict[int, asyncio.Lock] = {}
        self.group_recent_messages: dict[int, list[Message]] = {}
        self.group_last_processed_message_id: dict[int, int] = {}
        self.group_flush_tasks: dict[int, asyncio.Task] = {}
        if context_token_limit <= 0:
            raise ValueError("context_token_limit must be positive")
        if response_token_reserve < 0 or response_token_reserve >= context_token_limit:
            raise ValueError("response_token_reserve must be between 0 and context_token_limit")
        self.context_token_limit = context_token_limit
        self.response_token_reserve = response_token_reserve
        self.group_recent_window_size = 5
        self.group_flush_interval_seconds = 20
        self.max_input_image_bytes = 5_000_000
        self.max_private_media_items = 2
        self.max_group_media_items = 3

        self._bot_user_id: int | None = None
        self._bot_username: str = ""
        self._bot_meta_lock = asyncio.Lock()


        self.prompt = Prompt()

    async def _ensure_bot_meta(self) -> None:
        if self._bot_user_id is not None:
            return

        async with self._bot_meta_lock:
            if self._bot_user_id is not None:
                return
            me = await self.telegram_bot.get_me()
            self._bot_user_id = me.id
            self._bot_username = (me.username or "").lower()

    def _message_mentions_bot(self, message: Message) -> bool:
        if self._bot_user_id is None:
            return False

        entities = list(message.entities or []) + list(message.caption_entities or [])
        text = message.text or message.caption or ""

        for entity in entities:
            entity_type = getattr(entity.type, "value", str(entity.type)).lower()
            if entity_type == "text_mention" and getattr(getattr(entity, "user", None), "id", None) == self._bot_user_id:
                return True
            if entity_type == "mention" and self._bot_username and text:
                start = entity.offset
                end = entity.offset + entity.length
                mention_text = text[start:end].lower()
                if mention_text == f"@{self._bot_username}":
                    return True

        if self._bot_username:
            return f"@{self._bot_username}" in text.lower()
        return False

    def _message_replies_to_bot(self, message: Message) -> bool:
        if self._bot_user_id is None:
            return False
        reply_to = message.reply_to_message
        if reply_to is None or reply_to.from_user is None:
            return False
        return reply_to.from_user.id == self._bot_user_id

    async def _extract_message_image_part(self, message: Message) -> tuple[dict[str, Any] | None, str | None]:
        file_id: str | None = None
        mime_type: str | None = None
        media_tag: str | None = None

        if message.photo:
            largest_photo = message.photo[-1]
            file_id = largest_photo.file_id
            mime_type = "image/jpeg"
            media_tag = "photo"
        elif message.sticker:
            sticker = message.sticker
            if sticker.is_animated or sticker.is_video:
                return None, f"message_id={message.message_id} contains animated/video sticker, skipped for image understanding."
            file_id = sticker.file_id
            mime_type = "image/webp"
            media_tag = "sticker"
        elif message.document and (message.document.mime_type or "").startswith("image/"):
            file_id = message.document.file_id
            mime_type = message.document.mime_type
            media_tag = "image_document"
        else:
            return None, None

        try:
            telegram_file = await self.telegram_bot.get_file(file_id)
            if not telegram_file.file_path:
                return None, f"message_id={message.message_id} has {media_tag}, but file_path is missing."

            stream = io.BytesIO()
            await self.telegram_bot.download_file(telegram_file.file_path, destination=stream)
            data = stream.getvalue()
            if not data:
                return None, f"message_id={message.message_id} has empty {media_tag} file."
            if len(data) > self.max_input_image_bytes:
                return (
                    None,
                    f"message_id={message.message_id} {media_tag} too large ({len(data)} bytes), skipped.",
                )

            guessed_mime = mimetypes.guess_type(telegram_file.file_path)[0]
            final_mime = mime_type or guessed_mime or "image/jpeg"
            if not final_mime.startswith("image/"):
                return None, f"message_id={message.message_id} {media_tag} mime={final_mime}, not an image."

            encoded = base64.b64encode(data).decode("ascii")
            return {"type": "image_url", "image_url": {"url": f"data:{final_mime};base64,{encoded}"}}, None
        except Exception as e:
            logging.warning("Failed to prepare image input for message_id=%s: %s", message.message_id, e)
            return None, f"message_id={message.message_id} failed to load {media_tag}."

    async def _run_chat(
        self,
        messages_input: list[Message],
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages_input:
            return history

        anchor_message = messages_input[-1]
        ctx = ToolContext(bot=self.telegram_bot, message=anchor_message, superuser_id=self.superuser_id)
        executor = ToolExecutor(self.tools)

        is_group_chat = not _is_private_chat(anchor_message)
        if is_group_chat:
            await self._ensure_bot_meta()
            batch_items = [
                GroupBatchItem(
                    message_id=msg.message_id,
                    text=_message_text(msg),
                    date=msg.date,
                    from_user_full_name=getattr(msg.from_user, "full_name", "Unknown"),
                    from_user_id=getattr(msg.from_user, "id", 0),
                    mentions_bot=self._message_mentions_bot(msg),
                    replies_to_bot=self._message_replies_to_bot(msg),
                )
                for msg in messages_input
            ]
            selected_messages: list[Message] = []
            priority_items = [item for item in batch_items if item.replies_to_bot or item.mentions_bot]
            priority_items.sort(
                key=lambda item: (
                    1 if item.replies_to_bot else 0,
                    1 if item.mentions_bot else 0,
                    item.message_id,
                ),
                reverse=True,
            )
            priority_message_ids = [item.message_id for item in priority_items]
            priority_ids = set(priority_message_ids)
            selected_messages.extend([msg for msg in messages_input if msg.message_id in priority_ids])
            selected_messages.extend([msg for msg in reversed(messages_input) if msg.message_id not in priority_ids])

            image_parts: list[dict[str, Any]] = []
            media_notes: list[str] = []
            for msg in selected_messages:
                if len(image_parts) >= self.max_group_media_items:
                    break
                image_part, note = await self._extract_message_image_part(msg)
                if image_part:
                    image_parts.append(image_part)
                    media_notes.append(f"message_id={msg.message_id} includes image input for understanding.")
                elif note:
                    media_notes.append(note)
            user_payload = GroupBatchInput(
                chat_id=anchor_message.chat.id,
                batch_size=len(batch_items),
                priority_message_ids=priority_message_ids,
                messages=batch_items,
                media_notes=media_notes,
            ).model_dump_json()
            if image_parts:
                user_content: Any = [{"type": "text", "text": user_payload}, *image_parts]
            else:
                user_content = user_payload
        else:
            single = messages_input[0]
            image_parts: list[dict[str, Any]] = []
            media_notes: list[str] = []
            image_part, note = await self._extract_message_image_part(single)
            if image_part:
                image_parts.append(image_part)
                media_notes.append(f"message_id={single.message_id} includes image input for understanding.")
            elif note:
                media_notes.append(note)

            user_payload = BotInputMessage(
                chat_id=single.chat.id,
                text=_message_text(single),
                date=single.date,
                from_user_full_name=single.from_user.full_name,
                message_id=single.message_id,
                from_user_id=single.from_user.id,
                media_notes=media_notes,
            ).model_dump_json()
            if image_parts:
                user_content = [{"type": "text", "text": user_payload}, *image_parts[:self.max_private_media_items]]
            else:
                user_content = user_payload

        messages: list[dict] = [
            {
                "role": "system",
                "content": self.prompt.build(
                    PromptContext(
                        current_date=anchor_message.date,
                        user_full_name=anchor_message.from_user.full_name,
                        user_id=anchor_message.from_user.id,
                        chat_type=str(anchor_message.chat.type),
                        superuser_id=self.superuser_id,
                        available_tools=self.tools.schemas,
                    )
                ),
            },
            *history,
            {"role": "user", "content": user_content},
        ]

        logging.debug(
            "Initial messages for AI: total=%s history=%s user_content_type=%s",
            len(messages),
            len(history),
            type(user_content).__name__,
        )

        for _ in range(8):
            fixed_tokens = estimate_message_tokens(messages[0]) + estimate_tokens(self.tools.schemas)
            conversation_budget = self.context_token_limit - self.response_token_reserve - fixed_tokens
            compressed_messages, was_compressed = compress_history(
                messages[1:],
                available_tokens=conversation_budget,
            )
            if was_compressed:
                before_count = len(messages) - 1
                messages = [messages[0], *compressed_messages]
                logging.info(
                    "Compressed chat context: before=%s after=%s estimated_tokens=%s limit=%s",
                    before_count,
                    len(compressed_messages),
                    fixed_tokens + sum(estimate_message_tokens(message) for message in compressed_messages),
                    self.context_token_limit,
                )

            # noinspection PyTypeChecker
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools.schemas,
                tool_choice="auto",
            )

            m = response.choices[0].message
            tool_calls = getattr(m, "tool_calls", None)

            logging.info(
                "AI raw assistant content: %s | tool_calls=%s",
                (m.content or "").strip(),
                "yes" if tool_calls else "no",
            )

            if tool_calls:
                logging.info(
                    "AI decided to call tools: %s(%s)",
                    [tc.function.name for tc in tool_calls],
                    [tc.function.arguments for tc in tool_calls],
                )

                messages.append({
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                })

                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args_json = tc.function.arguments or "{}"

                    result = await executor.execute(ctx, tool_name, tool_args_json)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                continue

            messages.append({"role": "assistant", "content": m.content})

            return messages[1:]

        await anchor_message.answer("我这边有点卡 你再说一遍")
        logging.warning("Tool loop exceeded max turns")
        return messages[1:]

    async def _run_group_periodic_flush(self, chat_id: int) -> None:
        task = asyncio.current_task()
        try:
            while True:
                await asyncio.sleep(self.group_flush_interval_seconds)

                lock = self.chat_locks.setdefault(chat_id, asyncio.Lock())
                async with lock:
                    recent_messages = self.group_recent_messages.get(chat_id, [])
                    if not recent_messages:
                        continue

                    latest_message = recent_messages[-1]
                    last_processed_id = self.group_last_processed_message_id.get(chat_id)
                    if last_processed_id == latest_message.message_id:
                        continue

                    history = list(self.chat_histories.get(chat_id, []))
                    logging.info(
                        "Group periodic flush: chat_id=%s latest_message_id=%s batch_size=%s",
                        chat_id,
                        latest_message.message_id,
                        len(recent_messages),
                    )

                    batch = list(recent_messages)
                    updated_history = await self._run_chat(messages_input=batch, history=history)
                    self.chat_histories[chat_id] = updated_history
                    self.group_last_processed_message_id[chat_id] = latest_message.message_id
        except Exception:
            logging.exception("Group periodic flush task failed: chat_id=%s", chat_id)
        finally:
            existing = self.group_flush_tasks.get(chat_id)
            if existing is task:
                await self.group_flush_tasks.pop(chat_id, None)

    async def chat(self, message: Message, reset_history: bool = False) -> None:
        chat_id = message.chat.id
        lock = self.chat_locks.setdefault(chat_id, asyncio.Lock())

        async with lock:
            if reset_history:
                self.chat_histories.pop(chat_id, None)
                self.group_recent_messages.pop(chat_id, None)
                self.group_last_processed_message_id.pop(chat_id, None)
                flush_task = self.group_flush_tasks.pop(chat_id, None)
                if flush_task and not flush_task.done():
                    flush_task.cancel()

            history = list(self.chat_histories.get(chat_id, []))

            if _is_private_chat(message):
                updated_history = await self._run_chat(messages_input=[message], history=history)
                self.chat_histories[chat_id] = updated_history
                return

            await self._ensure_bot_meta()
            recent = self.group_recent_messages.setdefault(chat_id, [])
            recent.append(message)
            if len(recent) > self.group_recent_window_size:
                del recent[:-self.group_recent_window_size]
            logging.info(
                "Group message tracked: chat_id=%s latest_message_id=%s recent_size=%s",
                chat_id,
                message.message_id,
                len(recent),
            )

            flush_task = self.group_flush_tasks.get(chat_id)
            if flush_task is None or flush_task.done():
                self.group_flush_tasks[chat_id] = asyncio.create_task(self._run_group_periodic_flush(chat_id))
                logging.info(
                    "Group periodic flush task started: chat_id=%s interval=%ss",
                    chat_id,
                    self.group_flush_interval_seconds,
                )

    async def chat_without_history(self, message: Message) -> None:
        await self._run_chat(messages_input=[message], history=[])
