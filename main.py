import os
import io
import time
import math
import asyncio
import tempfile
import shutil
from typing import Dict, List, Optional
from datetime import datetime

from pyrogram import Client, filters, enums
from pyrogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
)

# Imaging / PDF
from PIL import Image
from PyPDF2 import PdfMerger, PdfReader
import numpy as np
import cv2
import fitz  # PyMuPDF
import subprocess

# Optional structural optimization
try:
    import pikepdf
except Exception:
    pikepdf = None

# -----------------------------
# Config from environment
# -----------------------------
API_ID = int(os.getenv("API_ID", "23907288"))
API_HASH = os.getenv("API_HASH", "f9a47570ed19aebf8eb0f0a5ec1111e5")
BOT_TOKEN = os.getenv("BOT_TOKEN", "8334418090:AAF0MjtAhQ9RzbyAxtHh6m-yXF6VrYfO-OM")

if not API_ID or not API_HASH or not BOT_TOKEN:
    raise SystemExit("Set API_ID, API_HASH, BOT_TOKEN environment variables.")

# -----------------------------
# App
# -----------------------------
app = Client("pdf_tool_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, in_memory=True)

# -----------------------------
# Per-user session (supports many users concurrently)
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
        self.collected_files: List[str] = []   # for merge / img2pdf
        self.target_file: Optional[str] = None # last received single PDF
        self.current_task: Optional[asyncio.Task] = None
        self.ocr_available = self._has_ocr()

    def _has_ocr(self) -> bool:
        try:
            subprocess.run(["ocrmypdf", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def cleanup(self):
        try:
            shutil.rmtree(self.workdir, ignore_errors=True)
        except Exception:
            pass

SESSIONS: Dict[int, Session] = {}

def get_session(uid: int) -> Session:
    if uid not in SESSIONS:
        SESSIONS[uid] = Session(uid)
    return SESSIONS[uid]

# -----------------------------
# Helpers
# -----------------------------
def human_size(num: float) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024:
            return f"{num:.2f}{unit}"
        num /= 1024
    return f"{num:.2f}PB"

def eta_text(start_t: float, cur: int, total: int) -> str:
    elapsed = max(0.001, time.time()-start_t)
    spd = cur/elapsed
    remain = (total-cur)/spd if spd>0 else 0
    return f"{human_size(spd)}/s • ETA {int(remain)}s"

def progress_cb_builder(status_msg: Message, action: str):
    start_t = time.time()
    last = {"t": 0}

    async def _cb(current: int, total: int):
        try:
            now = time.time()
            if current != total and now - last["t"] < 1.2:
                return
            last["t"] = now
            pct = (current/total*100) if total else 0
            bar_len = 20
            filled = int(bar_len * pct/100)
            bar = "█"*filled + "░"*(bar_len-filled)
            txt = f"{action}\n[{bar}] {pct:.1f}%\n{human_size(current)} / {human_size(total)}\n{eta_text(start_t, current, total)}"
            await status_msg.edit_text(txt)
        except Exception:
            pass
    return _cb

async def send_document_with_progress(chat_id: int, path: str, caption: str, reply_to: Optional[Message]=None):
    size = os.path.getsize(path)
    status = await app.send_message(chat_id, "⬆️ Preparing upload...")
    prog = progress_cb_builder(status, "⬆️ Uploading...")
    try:
        await app.send_document(
            chat_id,
            document=path,
            caption=caption,
            progress=prog,
            reply_to_message_id=reply_to.id if reply_to else None
        )
    finally:
        try: await status.delete()
        except Exception: pass

def ensure_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            PdfReader(f)
        return True
    except Exception:
        return False

def pil_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA","LA"):
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def images_to_pdf(image_paths: List[str], out_path: str):
    imgs = [pil_rgb(Image.open(p)) for p in image_paths]
    if not imgs: raise RuntimeError("No images")
    if len(imgs)==1:
        imgs[0].save(out_path, "PDF", resolution=200.0)
    else:
        imgs[0].save(out_path, "PDF", resolution=200.0, save_all=True, append_images=imgs[1:])

def deskew_image(cv_img):
    gray = cv_img if len(cv_img.shape)==2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thr = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr>0))
    angle = 0.0
    if coords.size>0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90+angle) if angle<-45 else -angle
    (h,w) = cv_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(cv_img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def clean_scan(cv_img):
    den = cv2.fastNlMeansDenoisingColored(cv_img, None, 5,5,7,21)
    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def save_temp_image(img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, "JPEG", quality=95)
    tmp.close()
    return tmp.name

# -----------------------------
# Smart Compression (non-blocking wrapper uses to_thread)
# -----------------------------
def compress_pdf_smart(input_path: str, output_path: str, progress_callback=None):
    """
    Smart: try structural optimize with pikepdf (keeps text/searchable).
    If no gain, fall back to light raster (image-heavy PDFs).
    """
    # 1) Structural (pikepdf) if available
    try:
        if pikepdf is not None:
            with pikepdf.open(input_path) as pdf:
                try:
                    pdf.remove_unreferenced_resources()
                except Exception:
                    pass
                pdf.save(output_path, linearize=True)
            orig = os.path.getsize(input_path)
            new = os.path.getsize(output_path)
            if orig>0 and (orig-new)/orig > 0.02:
                return  # Good enough shrink
    except Exception:
        # Will fall back
        pass

    # 2) Light raster fallback
    doc = fitz.open(input_path)
    total = len(doc)
    out = fitz.open()
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=110, alpha=False)
        rect = fitz.Rect(0,0,pix.width,pix.height)
        p = out.new_page(width=rect.width, height=rect.height)
        try:
            img_bytes = pix.tobytes("jpeg", quality=65)
        except TypeError:
            img_bytes = pix.tobytes("jpeg")
        p.insert_image(rect, stream=img_bytes)
        if progress_callback:
            progress_callback(idx, total)
    out.save(output_path, deflate=True)
    out.close()
    doc.close()

# -----------------------------
# Scan PDF (with live progress)
# -----------------------------
def pdf_scan(input_path: str, output_path: str, try_ocr: bool, progress_callback=None):
    doc = fitz.open(input_path)
    total = len(doc)
    imgs: List[Image.Image] = []
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        arr = deskew_image(arr)
        arr = clean_scan(arr)
        pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        imgs.append(pil_img)
        if progress_callback:
            progress_callback(idx, total)
    tmp_paths = [save_temp_image(i) for i in imgs]
    images_to_pdf(tmp_paths, output_path)

    if try_ocr:
        ocr_out = output_path.replace(".pdf", "_ocr.pdf")
        try:
            subprocess.run(
                ["ocrmypdf", "--skip-text", "--fast-web-view", output_path, ocr_out],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            shutil.move(ocr_out, output_path)
        except Exception:
            pass

# -----------------------------
# UI (keyboards)
# -----------------------------
def home_kb():
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
@app.on_message(filters.command(["start","help"]) & filters.private)
async def start_cmd(_, m: Message):
    s = get_session(m.from_user.id)
    await m.reply_text(
        "Hi! I’m your all-in-one PDF bot.\n\n"
        "Send a file or use buttons below. I show **download & upload progress** and keep working while big jobs run.\n\n"
        "• Compress → Shrink PDF (keeps text when possible)\n"
        "• Merge → Send multiple PDFs then /done\n"
        "• Images→PDF → Send images (order = send order), then /done\n"
        "• Scan → Clean & deskew (OCR if installed)\n"
        "• Rename → Send file, then send new name\n\n"
        "Use /cancel to stop current flow, /reset to clear workspace.",
        reply_markup=home_kb()
    )

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(_, m: Message):
    s = get_session(m.from_user.id)
    if s.current_task and not s.current_task.done():
        s.current_task.cancel()
    s.mode = Mode.IDLE
    s.collected_files.clear()
    s.target_file = None
    await m.reply_text("✅ Cancelled. Back to menu.", reply_markup=home_kb())

@app.on_message(filters.command("reset") & filters.private)
async def reset_cmd(_, m: Message):
    s = get_session(m.from_user.id)
    if s.current_task and not s.current_task.done():
        s.current_task.cancel()
    s.cleanup()
    SESSIONS.pop(m.from_user.id, None)
    await m.reply_text("🧹 Workspace cleared.", reply_markup=home_kb())

# -----------------------------
# Download helper
# -----------------------------
async def download_to(session: Session, msg: Message, status_msg: Message) -> Optional[str]:
    media = msg.document or msg.photo or msg.animation or msg.video or msg.sticker
    if not media: return None
    filename = getattr(media, "file_name", None)
    if not filename:
        ext = ".pdf" if msg.document and msg.document.mime_type=="application/pdf" else ".bin"
        filename = f"file_{msg.id}{ext}"
    out_path = os.path.join(session.workdir, filename)
    prog = progress_cb_builder(status_msg, "⬇️ Downloading...")
    path = await app.download_media(msg, file_name=out_path, progress=prog)
    return path

# -----------------------------
# Collecting files
# -----------------------------
@app.on_message((filters.document | filters.photo) & filters.private)
async def handle_files(_, m: Message):
    s = get_session(m.from_user.id)
    status = await m.reply_text("⬇️ Starting download...")
    try:
        path = await download_to(s, m, status)
    finally:
        try: await status.delete()
        except Exception: pass

    if not path or not os.path.exists(path):
        await m.reply_text("❌ Failed to download.")
        return

    if s.mode == Mode.MERGE:
        if not ensure_pdf(path):
            await m.reply_text("⚠️ Please send PDFs for merging.")
            return
        s.collected_files.append(path)
        await m.reply_text(f"📥 Added `{os.path.basename(path)}`. Total: **{len(s.collected_files)}**\nSend more, or /done.")
        return

    if s.mode == Mode.IMG2PDF:
        try:
            Image.open(path)
        except Exception:
            await m.reply_text("⚠️ Send images (JPG/PNG) for Images→PDF.")
            return
        s.collected_files.append(path)
        await m.reply_text(f"🖼️ Added image `{os.path.basename(path)}`. Total: **{len(s.collected_files)}**\nSend more, or /done.")
        return

    if s.mode == Mode.RENAME:
        s.target_file = path
        await m.reply_text(f"✏️ Got `{os.path.basename(path)}`.\nNow send the *new name with extension*, e.g. `myfile.pdf`.")
        return

    # Idle: propose quick actions
    if ensure_pdf(path):
        s.target_file = path
        await m.reply_text(
            f"📄 Received `{os.path.basename(path)}` ({human_size(os.path.getsize(path))}). Choose:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🗜️ Compress", callback_data="compress_go")],
                [InlineKeyboardButton("🖨️ Scan (clean/deskew)", callback_data="scan_go")],
                [InlineKeyboardButton("✏️ Rename", callback_data="rename")],
                [InlineKeyboardButton("🏠 Home", callback_data="help")]
            ])
        )
    else:
        # If image while idle, auto-switch to img2pdf
        try:
            Image.open(path)
            s.collected_files = [path]
            s.mode = Mode.IMG2PDF
            await m.reply_text("🖼️ Image received. **Images→PDF** mode enabled.\nSend more images, or /done.")
        except Exception:
            await m.reply_text("File received. Pick an action.", reply_markup=home_kb())

# -----------------------------
# /done for merge / img2pdf
# -----------------------------
@app.on_message(filters.command("done") & filters.private)
async def done_cmd(_, m: Message):
    s = get_session(m.from_user.id)

    if s.mode == Mode.MERGE:
        if len(s.collected_files) < 2:
            await m.reply_text("Send at least **2 PDFs** before /done.")
            return
        out = os.path.join(s.workdir, f"merged_{int(time.time())}.pdf")
        await m.reply_text("🔧 Merging PDFs...")
        try:
            merger = PdfMerger()
            for p in s.collected_files:
                merger.append(p)
            merger.write(out)
            merger.close()
            await send_document_with_progress(m.chat.id, out, "✅ Merged PDF", reply_to=m)
        except Exception as e:
            await m.reply_text(f"❌ Merge failed: `{e}`")
        finally:
            s.collected_files.clear()
            s.mode = Mode.IDLE
        return

    if s.mode == Mode.IMG2PDF:
        if len(s.collected_files) < 1:
            await m.reply_text("Send at least **1 image** before /done.")
            return
        out = os.path.join(s.workdir, f"images_{int(time.time())}.pdf")
        await m.reply_text("🧯 Converting images to PDF...")
        try:
            images_to_pdf(s.collected_files, out)
            await send_document_with_progress(m.chat.id, out, "✅ Images → PDF", reply_to=m)
        except Exception as e:
            await m.reply_text(f"❌ Conversion failed: `{e}`")
        finally:
            s.collected_files.clear()
            s.mode = Mode.IDLE
        return

    await m.reply_text("Nothing to finish. Pick an action.", reply_markup=home_kb())

# -----------------------------
# Unified callback handler (no conflicts)
# -----------------------------
# -----------------------------
# -----------------------------
@app.on_callback_query()
async def all_callbacks(_: Client, cq: CallbackQuery):
    await cq.answer()
    s = get_session(cq.from_user.id)
    data = cq.data

    # BASIC MENU -------------------------------------------------------------
    if data == "help":
        await cq.message.edit_text(
            "📖 **How to use**\n\n"
            "• Compress → Send a PDF\n"
            "• Merge → Send multiple PDFs, then /done\n"
            "• Images→PDF → Send images, then /done\n"
            "• Scan → Send a PDF (clean & deskew)\n"
            "• Rename → Send file, then a new name\n\n"
            "Use /cancel anytime.",
            reply_markup=home_kb()
        )
        return

    if data == "reset":
        if s.current_task and not s.current_task.done():
            s.current_task.cancel()
        s.cleanup()
        SESSIONS.pop(cq.from_user.id, None)
        await cq.message.edit_text("🧹 Workspace cleared.", reply_markup=home_kb())
        return

    # START MODES -------------------------------------------------------------
    if data == "compress":
        s.mode = Mode.IDLE
        s.target_file = None
        await cq.message.edit_text("🗜️ Send the **PDF** you want to compress.")
        return

    if data == "merge":
        s.mode = Mode.MERGE
        s.collected_files.clear()
        await cq.message.edit_text("➕ Send **multiple PDFs** in order. Send **/done** when finished.")
        return

    if data == "img2pdf":
        s.mode = Mode.IMG2PDF
        s.collected_files.clear()
        await cq.message.edit_text("🖼️ Send **images** in order. Send **/done** when finished.")
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

    # PER-FILE QUICK ACTIONS -------------------------------------------------
    if data == "compress_go":
        if not s.target_file or not os.path.exists(s.target_file):
            await cq.message.reply_text("⚠️ Please send a PDF first.")
            return

        in_path = s.target_file
        out_path = os.path.join(s.workdir, f"compressed_{int(time.time())}.pdf")
        msg = await cq.message.edit_text("🗜️ Starting compression...")

        main_loop = asyncio.get_running_loop()  # capture loop before thread

        async def run_compress():
            last_update = 0

            def progress_callback(cur, total):
                nonlocal last_update
                now = time.time()
                if now - last_update > 1:
                    last_update = now

                    async def _edit():
                        try:
                            await msg.edit_text(f"🗜️ Compressing... Page {cur}/{total}")
                        except Exception:
                            pass

                    asyncio.run_coroutine_threadsafe(_edit(), main_loop)

            try:
                before = os.path.getsize(in_path)
                await asyncio.to_thread(compress_pdf_smart, in_path, out_path, progress_callback)
                after = os.path.getsize(out_path)
                red = (1 - after / before) * 100 if before else 0
                await msg.edit_text("✅ Compression complete! Uploading...")
                await send_document_with_progress(
                    cq.message.chat.id,
                    out_path,
                    f"✅ Compressed (~{red:.1f}% smaller)"
                )
            except asyncio.CancelledError:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                await msg.edit_text("❌ Compression cancelled.")
            except Exception as e:
                await msg.edit_text(f"❌ Compression failed: `{e}`")
            finally:
                s.mode = Mode.IDLE
                s.target_file = None
                s.current_task = None

        if s.current_task and not s.current_task.done():
            await cq.message.reply_text("⚠️ Another job is running. Send /cancel to stop it first.")
            return

        s.current_task = asyncio.create_task(run_compress())
        return

    if data == "scan_go":
        if not s.target_file or not os.path.exists(s.target_file):
            await cq.message.reply_text("⚠️ Please send a PDF first.")
            return

        in_path = s.target_file
        out_path = os.path.join(s.workdir, f"scanned_{int(time.time())}.pdf")
        msg = await cq.message.edit_text("🖨️ Starting scan...")

        main_loop = asyncio.get_running_loop()

        async def run_scan():
            last_update = 0

            def progress_callback(cur, total):
                nonlocal last_update
                now = time.time()
                if now - last_update > 1:
                    last_update = now

                    async def _edit():
                        try:
                            await msg.edit_text(f"🖨️ Scanning... Page {cur}/{total}")
                        except Exception:
                            pass

                    asyncio.run_coroutine_threadsafe(_edit(), main_loop)

            try:
                await asyncio.to_thread(pdf_scan, in_path, out_path, s.ocr_available, progress_callback)
                note = " (with OCR)" if s.ocr_available else ""
                await msg.edit_text("✅ Scan complete! Uploading...")
                await send_document_with_progress(
                    cq.message.chat.id,
                    out_path,
                    f"✅ Scanned{note}"
                )
            except asyncio.CancelledError:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                await msg.edit_text("❌ Scan cancelled.")
            except Exception as e:
                await msg.edit_text(f"❌ Scan failed: `{e}`")
            finally:
                s.mode = Mode.IDLE
                s.target_file = None
                s.current_task = None

        if s.current_task and not s.current_task.done():
            await cq.message.reply_text("⚠️ Another job is running. Send /cancel to stop it first.")
            return

        s.current_task = asyncio.create_task(run_scan())
        return

    await cq.answer("Unknown action.", show_alert=True)
@app.on_message(filters.text & filters.private)
async def handle_text(_, m: Message):
    s = get_session(m.from_user.id)
    if m.text.strip() in ("/done","/reset"):  # handled by commands
        return

    if s.mode == Mode.RENAME and s.target_file:
        new_name = m.text.strip()
        if not new_name or "/" in new_name or "\\" in new_name:
            await m.reply_text("❌ Invalid name. Avoid slashes. Include an extension, e.g. `name.pdf`.")
            return
        new_path = os.path.join(s.workdir, new_name)
        try:
            os.rename(s.target_file, new_path)
            await send_document_with_progress(m.chat.id, new_path, "✅ Renamed", reply_to=m)
        except Exception as e:
            await m.reply_text(f"❌ Rename failed: `{e}`")
        finally:
            s.mode = Mode.IDLE
            s.target_file = None
        return

    # Otherwise
    if s.mode == Mode.IDLE:
        await m.reply_text("Pick an action or send a file.", reply_markup=home_kb())
    else:
        await m.reply_text("Waiting for files… or use /done.")

# -----------------------------
# Cleanup
# -----------------------------
import atexit
@atexit.register
def _cleanup_all():
    for s in list(SESSIONS.values()):
        try:
            if s.current_task and not s.current_task.done():
                s.current_task.cancel()
        except Exception:
            pass
        s.cleanup()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("Bot running…")
    app.run()
