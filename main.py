import os
import io
import math
import time
import asyncio
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pyrogram import Client, filters, enums
from pyrogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, InputMediaDocument
)

# PDF/Imaging libs
from PIL import Image
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import pikepdf  # for compression
import fitz  # PyMuPDF
import cv2
import numpy as np
import subprocess

# -----------------------------
# Config via environment
# -----------------------------
API_ID = int(os.getenv("API_ID", "23907288"))
API_HASH = os.getenv("API_HASH", "f9a47570ed19aebf8eb0f0a5ec1111e5")
BOT_TOKEN = os.getenv("BOT_TOKEN", "8334418090:AAF0MjtAhQ9RzbyAxtHh6m-yXF6VrYfO-OM")

if not all([API_ID, API_HASH, BOT_TOKEN]):
    raise SystemExit("Please set API_ID, API_HASH, BOT_TOKEN as environment variables.")

# -----------------------------
# App
# -----------------------------
app = Client("pdf_tool_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, in_memory=True)

# -----------------------------
# In-memory user sessions
# -----------------------------
class Mode:
    IDLE = "IDLE"
    MERGE = "MERGE"
    IMG2PDF = "IMG2PDF"
    RENAME = "RENAME"

class Session:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.mode = Mode.IDLE
        self.workdir = tempfile.mkdtemp(prefix=f"pdfbot_{user_id}_")
        self.collected_files: List[str] = []  # for merge / img2pdf
        self.target_file: Optional[str] = None  # for compress/scan/rename target
        self.ocr_available = self._check_ocr()
    def cleanup(self):
        try:
            shutil.rmtree(self.workdir, ignore_errors=True)
        except Exception:
            pass
    def _check_ocr(self) -> bool:
        # detect presence of ocrmypdf
        try:
            subprocess.run(["ocrmypdf", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

SESSIONS: Dict[int, Session] = {}

def get_session(user_id: int) -> Session:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = Session(user_id)
    return SESSIONS[user_id]

# -----------------------------
# Utilities
# -----------------------------
def human_size(num: float) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024.0:
            return f"{num:.2f}{unit}"
        num /= 1024.0
    return f"{num:.2f}PB"

def eta_text(start_t: float, current: int, total: int) -> str:
    now = time.time()
    elapsed = max(0.001, now - start_t)
    speed = current / elapsed
    if speed > 0:
        remaining = (total - current) / speed
    else:
        remaining = 0
    return f"{human_size(speed)}/s • ETA {int(remaining)}s"

def progress_cb_builder(message: Message, action: str):
    start_t = time.time()
    last_update = {"t": 0}  # throttle edits

    async def _progress(current: int, total: int):
        try:
            now = time.time()
            if now - last_update["t"] < 1.5 and current != total:
                return
            last_update["t"] = now
            pct = (current / total) * 100 if total else 0
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            text = (
                f"{action}\n"
                f"[{bar}] {pct:.1f}%\n"
                f"{human_size(current)} / {human_size(total)}\n"
                f"{eta_text(start_t, current, total)}"
            )
            await message.edit_text(text)
        except Exception:
            pass
    return _progress

async def download_to(session: Session, msg: Message, status_msg: Message) -> Optional[str]:
    # Pick best media
    media = msg.document or msg.photo or msg.animation or msg.video or msg.sticker
    if not media:
        return None
    filename = getattr(media, "file_name", None)
    if not filename:
        # Create sensible name
        ext = ".pdf" if msg.document and msg.document.mime_type == "application/pdf" else ".bin"
        filename = f"file_{msg.id}{ext}"
    out_path = os.path.join(session.workdir, filename)
    prog = progress_cb_builder(status_msg, "⬇️ Downloading...")
    path = await app.download_media(msg, file_name=out_path, progress=prog)
    return path

def ensure_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            PdfReader(f)
        return True
    except Exception:
        return False

def compress_pdf(input_path: str, output_path: str, quality: int = 75):
    """
    Compress PDF by rasterizing pages and saving as images.
    Works across all PyMuPDF versions (1.20+).
    """
    try:
        doc = fitz.open(input_path)
        new_doc = fitz.open()

        for page in doc:
            pix = page.get_pixmap(dpi=120, alpha=False)

            # Create a new page same size
            page_rect = fitz.Rect(0, 0, pix.width, pix.height)
            page_pdf = new_doc.new_page(width=page_rect.width, height=page_rect.height)

            # Handle version differences (PyMuPDF >=1.24 removed 'quality' param)
            try:
                img_bytes = pix.tobytes("jpeg", quality=quality)
            except TypeError:
                # fallback for newer PyMuPDF versions (no quality argument)
                img_bytes = pix.tobytes("jpeg")

            page_pdf.insert_image(page_rect, stream=img_bytes)

        # Save with deflate compression
        new_doc.save(output_path, deflate=True)
        new_doc.close()
        doc.close()

    except Exception as e:
        raise RuntimeError(f"Compression failed: {e}")

def merge_pdfs(paths: List[str], output_path: str):
    merger = PdfMerger()
    for p in paths:
        merger.append(p)
    merger.write(output_path)
    merger.close()

def pil_save_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def images_to_pdf(image_paths: List[str], output_path: str):
    images: List[Image.Image] = []
    for p in image_paths:
        img = Image.open(p)
        images.append(pil_save_rgb(img))
    if not images:
        raise RuntimeError("No images.")
    if len(images) == 1:
        images[0].save(output_path, "PDF", resolution=200.0)
    else:
        images[0].save(output_path, "PDF", resolution=200.0, save_all=True, append_images=images[1:])

def deskew_image(cv_img: np.ndarray) -> np.ndarray:
    # Convert to gray and threshold
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def clean_scan(cv_img: np.ndarray) -> np.ndarray:
    # Gentle denoise + adaptive threshold to mimic "scanner" look
    den = cv2.fastNlMeansDenoisingColored(cv_img, None, 5, 5, 7, 21)
    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def pdf_scan(input_path: str, output_path: str, try_ocr: bool, progress_callback=None):
    """
    Scans PDF (deskew + clean) and optionally runs OCR.
    Runs synchronously but supports progress updates.
    """
    doc = fitz.open(input_path)
    total_pages = len(doc)
    page_images = []

    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        # clean + deskew
        img = deskew_image(img)
        img = clean_scan(img)
        page_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        # send live progress
        if progress_callback:
            progress_callback(idx, total_pages)

    # Save scanned pages as new PDF
    temp_images = [save_temp_image(p) for p in page_images]
    images_to_pdf(temp_images, output_path)

    # Run OCR if available
    if try_ocr:
        ocr_out = output_path.replace(".pdf", "_ocr.pdf")
        try:
            subprocess.run(["ocrmypdf", "--skip-text", "--fast-web-view", output_path, ocr_out],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.move(ocr_out, output_path)
        except Exception as e:
            print("OCR skipped:", e)
def save_temp_image(img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, "JPEG", quality=95)
    tmp.close()
    return tmp.name

async def send_document_with_progress(chat_id: int, path: str, caption: str, reply_to: Optional[Message] = None):
    size = os.path.getsize(path)
    status = await app.send_message(chat_id, "⬆️ Preparing upload...")
    prog = progress_cb_builder(status, "⬆️ Uploading...")
    try:
        await app.send_document(
            chat_id,
            document=path,
            caption=caption,
            progress=prog,
            progress_args=(),
            reply_to_message_id=reply_to.id if reply_to else None,
        )
    finally:
        try:
            await status.delete()
        except Exception:
            pass

# -----------------------------
# Keyboards
# -----------------------------
def home_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗜️ Compress PDF", callback_data="compress"),
         InlineKeyboardButton("➕ Merge PDFs", callback_data="merge")],
        [InlineKeyboardButton("🖼️ Images → PDF", callback_data="img2pdf"),
         InlineKeyboardButton("🖨️ Scan PDF", callback_data="scan")],
        [InlineKeyboardButton("✏️ Rename", callback_data="rename"),
         InlineKeyboardButton("🧹 Reset", callback_data="reset")],
        [InlineKeyboardButton("❓ Help", callback_data="help")]
    ])

# -----------------------------
# Commands
# -----------------------------
@app.on_message(filters.command(["start", "help"]) & filters.private)
async def start_cmd(client: Client, message: Message):
    s = get_session(message.from_user.id)
    await message.reply_text(
        "Hi! I’m your all-in-one PDF bot.\n\n"
        "Send a PDF or image, or pick an action below. I show **download & upload progress**.\n\n"
        "• Compress: Reduce PDF size\n"
        "• Merge: Collect multiple PDFs then /done\n"
        "• Images→PDF: Send multiple images then /done\n"
        "• Scan: Clean & deskew pages (optional OCR if installed)\n"
        "• Rename: Change a file name\n\n"
        "Use /cancel to stop the current flow, /reset to clear workspace.",
        reply_markup=home_kb()
    )

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client: Client, message: Message):
    s = get_session(message.from_user.id)
    s.mode = Mode.IDLE
    s.collected_files.clear()
    s.target_file = None
    await message.reply_text("✅ Cancelled. You're back at the main menu.", reply_markup=home_kb())

@app.on_message(filters.command("reset") & filters.private)
async def reset_cmd(client: Client, message: Message):
    s = get_session(message.from_user.id)
    s.cleanup()
    SESSIONS.pop(message.from_user.id, None)
    await message.reply_text("🧹 Workspace cleared.", reply_markup=home_kb())

# -----------------------------
# Callback actions
# -----------------------------
# -----------------------------
# SINGLE CALLBACK HANDLER (fixes compress/scan)
# -----------------------------
@app.on_callback_query()
async def all_callbacks(client: Client, cq: CallbackQuery):
    """Unified handler for all callback buttons."""
    await cq.answer()
    s = get_session(cq.from_user.id)
    data = cq.data

    # ---------- BASIC MENU ----------
    if data == "help":
        await cq.message.edit_text(
            "📖 **How to use this bot:**\n\n"
            "• Compress → Send a PDF\n"
            "• Merge → Send multiple PDFs, then /done\n"
            "• Images→PDF → Send multiple images, then /done\n"
            "• Scan → Send a PDF to clean & deskew\n"
            "• Rename → Send a file, then a new name\n\n"
            "Use /cancel anytime.",
            reply_markup=home_kb()
        )
        return

    if data == "reset":
        s.cleanup()
        SESSIONS.pop(cq.from_user.id, None)
        await cq.message.edit_text("🧹 Workspace cleared.", reply_markup=home_kb())
        return

    # ---------- START NEW MODES ----------
    if data == "compress":
        s.mode = Mode.IDLE
        s.target_file = None
        await cq.message.edit_text("🗜️ Send me the **PDF** you want to compress.")
        return

    if data == "merge":
        s.mode = Mode.MERGE
        s.collected_files.clear()
        await cq.message.edit_text("➕ Send **multiple PDFs** to merge (in order). When finished, send /done.")
        return

    if data == "img2pdf":
        s.mode = Mode.IMG2PDF
        s.collected_files.clear()
        await cq.message.edit_text("🖼️ Send **images** in order. When finished, send /done.")
        return

    if data == "scan":
        s.mode = Mode.IDLE
        s.target_file = None
        await cq.message.edit_text("🖨️ Send the **PDF** you want to scan.")
        return

    if data == "rename":
        s.mode = Mode.RENAME
        s.target_file = None
        await cq.message.edit_text("✏️ Send the **file** you want to rename.")
        return

    # ---------- PER-FILE ACTIONS ----------
    if data == "compress_go":
        if not s.target_file or not os.path.exists(s.target_file):
            await cq.message.reply_text("⚠️ Please send a PDF first.")
            return
        in_path = s.target_file
        out_path = os.path.join(s.workdir, f"compressed_{int(time.time())}.pdf")
        await cq.message.edit_text("🗜️ Compressing your PDF...\nPlease wait ⏳")

        try:
            before = os.path.getsize(in_path)
            compress_pdf(in_path, out_path)
            after = os.path.getsize(out_path)
            reduced = (1 - after / before) * 100 if before else 0
            await send_document_with_progress(
                cq.message.chat.id,
                out_path,
                f"✅ Compression complete!\nReduced by **{reduced:.1f}%**",
                reply_to=cq.message
            )
        except Exception as e:
            await cq.message.reply_text(f"❌ Compression failed: `{e}`")
        finally:
            s.mode = Mode.IDLE
            s.target_file = None
        return

    if data == "scan_go":
        if not s.target_file or not os.path.exists(s.target_file):
            await cq.message.reply_text("⚠️ Please send a PDF first.")
            return

        in_path = s.target_file
        out_path = os.path.join(s.workdir, f"scanned_{int(time.time())}.pdf")
        msg = await cq.message.edit_text("🖨️ Starting scan...")

        async def run_scan():
            last_update = 0

            def progress_callback(current, total):
                nonlocal last_update
                now = time.time()
                if now - last_update > 1:  # update every ~1 s
                    last_update = now
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        msg.edit_text(f"🖨️ Scanning... Page {current}/{total}")
                    )

            try:
                # run heavy work in background thread
                await asyncio.to_thread(
                    pdf_scan, in_path, out_path, s.ocr_available, progress_callback
                )
                note = " (with OCR)" if s.ocr_available else ""
                await msg.edit_text("✅ Scan complete! Uploading...")
                await send_document_with_progress(
                    cq.message.chat.id,
                    out_path,
                    f"✅ Scanned{note}",
                    reply_to=cq.message
                )
            except Exception as e:
                await msg.edit_text(f"❌ Scan failed: `{e}`")
            finally:
                s.mode = Mode.IDLE
                s.target_file = None

        # launch scan in background so bot stays responsive
        asyncio.create_task(run_scan())
        return

    # ---------- FALLBACK ----------
    await cq.answer("Unknown button clicked.", show_alert=True)
# -----------------------------
# /done handler
# -----------------------------
@app.on_message(filters.command("done") & filters.private)
async def done_cmd(client: Client, message: Message):
    s = get_session(message.from_user.id)

    if s.mode == Mode.MERGE:
        if len(s.collected_files) < 2:
            await message.reply_text("Please send at least **2 PDFs** before /done.")
            return
        out_path = os.path.join(s.workdir, f"merged_{int(time.time())}.pdf")
        await message.reply_text("🔧 Merging PDFs...")
        try:
            merge_pdfs(s.collected_files, out_path)
            await send_document_with_progress(message.chat.id, out_path, "✅ Merged PDF", reply_to=message)
        except Exception as e:
            await message.reply_text(f"❌ Merge failed: {e}")
        finally:
            s.collected_files.clear()
            s.mode = Mode.IDLE
        return

    if s.mode == Mode.IMG2PDF:
        if len(s.collected_files) < 1:
            await message.reply_text("Please send at least **1 image** before /done.")
            return
        out_path = os.path.join(s.workdir, f"images_{int(time.time())}.pdf")
        await message.reply_text("🧯 Converting images to PDF...")
        try:
            images_to_pdf(s.collected_files, out_path)
            await send_document_with_progress(message.chat.id, out_path, "✅ Images → PDF", reply_to=message)
        except Exception as e:
            await message.reply_text(f"❌ Conversion failed: {e}")
        finally:
            s.collected_files.clear()
            s.mode = Mode.IDLE
        return

    await message.reply_text("Nothing to finish. Pick an action first.", reply_markup=home_kb())

# -----------------------------
# File intake (documents/photos)
# -----------------------------
@ app.on_message((filters.document | filters.photo) & filters.private)
async def handle_files(client: Client, message: Message):
    s = get_session(message.from_user.id)

    # Start a status message for progress
    status = await message.reply_text("⬇️ Starting download...")

    try:
        path = await download_to(s, message, status_msg=status)
        if not path or not os.path.exists(path):
            await message.reply_text("❌ Couldn't download the file.")
            return
    finally:
        try:
            await status.delete()
        except Exception:
            pass

    # Route depending on mode
    if s.mode == Mode.MERGE:
        if not ensure_pdf(path):
            await message.reply_text("⚠️ That’s not a PDF. Please send PDFs for merging.")
            return
        s.collected_files.append(path)
        await message.reply_text(f"📥 Added: `{os.path.basename(path)}`\nTotal: **{len(s.collected_files)}**\nSend more or /done.", quote=True)
        return

    if s.mode == Mode.IMG2PDF:
        # Accept images or PDFs with images? Here: only images.
        try:
            Image.open(path)  # validate it's an image
            s.collected_files.append(path)
            await message.reply_text(f"🖼️ Added image `{os.path.basename(path)}`\nTotal: **{len(s.collected_files)}**\nSend more or /done.")
        except Exception:
            await message.reply_text("⚠️ Please send images (JPG/PNG) for Images→PDF.")
        return

    if s.mode == Mode.RENAME:
        s.target_file = path
        base = os.path.basename(path)
        await message.reply_text(f"✏️ Received `{base}`.\nNow send the *new name with extension*, e.g. `myfile.pdf`.", quote=True)
        return

    # Default (IDLE): decide by filetype or previously pressed callback
    # If a PDF arrives while idle, show quick actions
    if ensure_pdf(path):
        s.target_file = path
        await message.reply_text(
            f"📄 Received `{os.path.basename(path)}` ({human_size(os.path.getsize(path))}). Choose an action:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🗜️ Compress", callback_data=f"compress_go")],
                [InlineKeyboardButton("🖨️ Scan (clean/deskew)", callback_data=f"scan_go")],
                [InlineKeyboardButton("✏️ Rename", callback_data=f"rename")],
                [InlineKeyboardButton("🏠 Home", callback_data="help")]
            ])
        )
    else:
        # If it's an image while idle, suggest Images→PDF
        try:
            Image.open(path)
            s.collected_files = [path]
            s.mode = Mode.IMG2PDF
            await message.reply_text("🖼️ Image received. I’ve switched to **Images→PDF** mode.\nSend more images, or /done when finished.")
        except Exception:
            await message.reply_text("File received, but I’m not sure what to do. Pick an action.", reply_markup=home_kb())

# Quick action callbacks for idle PDFs
# -----------------------------

@app.on_callback_query(filters.regex("compress_go"))
async def handle_compress_callback(client: Client, cq: CallbackQuery):
    """Handle inline button click for Compress"""
    await cq.answer()  # acknowledge button press immediately
    user_id = cq.from_user.id
    session = get_session(user_id)

    if not session.target_file or not os.path.exists(session.target_file):
        await cq.message.reply_text("⚠️ Please send a PDF first.")
        return

    input_pdf = session.target_file
    output_pdf = os.path.join(session.workdir, f"compressed_{int(time.time())}.pdf")
    await cq.message.edit_text("🗜️ Compressing your PDF...\nPlease wait ⏳")

    try:
        before = os.path.getsize(input_pdf)
        compress_pdf(input_pdf, output_pdf)
        after = os.path.getsize(output_pdf)
        reduced = (1 - after / before) * 100 if before else 0

        await send_document_with_progress(
            cq.message.chat.id,
            output_pdf,
            f"✅ Compression complete! Reduced by **{reduced:.1f}%**",
            reply_to=cq.message
        )
    except Exception as e:
        await cq.message.reply_text(f"❌ Compression failed: `{e}`")
        print(f"[compress_go ERROR] {e}")
    finally:
        session.mode = Mode.IDLE
        session.target_file = None


@app.on_callback_query(filters.regex("scan_go"))
async def handle_scan_callback(client: Client, cq: CallbackQuery):
    """Handle inline button click for Scan"""
    await cq.answer()
    user_id = cq.from_user.id
    session = get_session(user_id)

    if not session.target_file or not os.path.exists(session.target_file):
        await cq.message.reply_text("⚠️ Please send a PDF first.")
        return

    input_pdf = session.target_file
    output_pdf = os.path.join(session.workdir, f"scanned_{int(time.time())}.pdf")
    await cq.message.edit_text("🖨️ Scanning your PDF...\nCleaning and deskewing pages ⚙️")

    try:
        pdf_scan(input_pdf, output_pdf, try_ocr=session.ocr_available)
        msg = "✅ Scan complete!"
        if session.ocr_available:
            msg += " (with OCR text recognition)"
        await send_document_with_progress(cq.message.chat.id, output_pdf, msg, reply_to=cq.message)
    except Exception as e:
        await cq.message.reply_text(f"❌ Scanning failed: `{e}`")
        print(f"[scan_go ERROR] {e}")
    finally:
        session.mode = Mode.IDLE
        session.target_file = None
# -----------------------------
# Rename text handler
# -----------------------------
@ app.on_message(filters.text & filters.private)
async def handle_text(client: Client, message: Message):
    s = get_session(message.from_user.id)

    if message.text.strip() == "/done":
        # handled by /done command already; ignore here for UX
        return

    if message.text.strip() == "/reset":
        await reset_cmd(client, message)
        return

    if s.mode == Mode.RENAME and s.target_file:
        new_name = message.text.strip()
        if not new_name or "/" in new_name or "\\" in new_name:
            await message.reply_text("❌ Invalid name. Avoid slashes. Include an extension, e.g. `name.pdf`.")
            return
        new_path = os.path.join(s.workdir, new_name)
        try:
            os.rename(s.target_file, new_path)
            await send_document_with_progress(message.chat.id, new_path, "✅ Renamed", reply_to=message)
        except Exception as e:
            await message.reply_text(f"❌ Rename failed: {e}")
        finally:
            s.mode = Mode.IDLE
            s.target_file = None
        return

    # If user types random text while idle
    if s.mode == Mode.IDLE:
        await message.reply_text("Pick an action below or send a file.", reply_markup=home_kb())
    else:
        await message.reply_text("Waiting for files… or use /done.", quote=True)

# -----------------------------
# Graceful shutdown cleanup
# -----------------------------
import atexit
@atexit.register
def cleanup_all():
    for s in list(SESSIONS.values()):
        s.cleanup()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("Bot is running...")
    app.run()
