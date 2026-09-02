"""
AutoSplit Studio docker.

Open a flat character image, click "Split Character": the docker exports the
active document, runs your SAM3 AutoSplit workflow in ComfyUI (on a background
thread so Krita stays responsive), and imports each part as a named, positioned
layer grouped into a tidy folder. "Export to Spine" writes a Spine 4.3.22 project
from the current split.

Edit the CONFIG block below if any path differs on your machine.
"""
import os
import tempfile
import traceback

from krita import Krita, DockWidget, InfoObject
from PyQt5.QtCore import QByteArray, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QPlainTextEdit, QLineEdit)

from . import comfy_client
from . import spine_export
from . import restyle

# ============================== CONFIG ==============================
# Set AUTOSPLIT_PROJECT to this repo's folder and AUTOSPLIT_COMFY_ROOT to your
# ComfyUI install; everything else is derived. Or override any single value
# with its own environment variable. Nothing here is machine-specific.
PROJECT_DIR = os.environ.get(
    "AUTOSPLIT_PROJECT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
COMFY_ROOT = os.environ.get(
    "AUTOSPLIT_COMFY_ROOT",
    os.path.join(os.path.dirname(PROJECT_DIR), "ComfyUI-Easy-Install", "ComfyUI"))

COMFY_URL = os.environ.get("AUTOSPLIT_COMFY_URL", "http://127.0.0.1:8188")
WORKFLOW_API = os.environ.get(
    "AUTOSPLIT_WORKFLOW",
    os.path.join(PROJECT_DIR, "2D_Character_AutoSplit_SAM3.json"))
COMFY_OUTPUT_DIR = os.environ.get(
    "AUTOSPLIT_COMFY_OUTPUT", os.path.join(COMFY_ROOT, "output"))
PART_SUBDIRS = ["split_parts_sam3", "split_parts_sam3_facial"]
SPINE_OUT_DIR = os.environ.get(
    "AUTOSPLIT_OUT", os.path.join(PROJECT_DIR, "spine_export"))
SPINE_VERSION = "4.3.22"
# ====================================================================


class SplitWorker(QThread):
    """Runs the ComfyUI split off the UI thread so Krita doesn't freeze."""
    progress = pyqtSignal(str)
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, comfy_url, workflow_api, image_path):
        super().__init__()
        self._url = comfy_url
        self._wf = workflow_api
        self._img = image_path

    def run(self):
        try:
            comfy_client.run_split(self._url, self._wf, self._img,
                                   progress=lambda m: self.progress.emit(str(m)))
            self.done.emit()
        except Exception as e:
            self.failed.emit("%s\n%s" % (e, traceback.format_exc()))


class RestyleWorker(QThread):
    """Runs the img2img restyle off the UI thread."""
    progress = pyqtSignal(str)
    done = pyqtSignal(str)      # restyled image path
    failed = pyqtSignal(str)

    def __init__(self, comfy_url, image_path, prompt, output_dir):
        super().__init__()
        self._url = comfy_url
        self._img = image_path
        self._prompt = prompt
        self._out = output_dir

    def run(self):
        try:
            path = restyle.run_restyle(self._url, self._img, self._prompt, self._out,
                                       progress=lambda m: self.progress.emit(str(m)))
            self.done.emit(path)
        except Exception as e:
            self.failed.emit("%s\n%s" % (e, traceback.format_exc()))


class AutoSplitDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoSplit Studio")
        self._worker = None
        self._restyle_worker = None

        root = QWidget()
        layout = QVBoxLayout()
        self.info = QLabel("Open a flat character image, then Split.")
        self.info.setWordWrap(True)
        self.btn = QPushButton("Split Character")
        self.btn.clicked.connect(self.on_split)
        self.btn_spine = QPushButton("Export to Spine")
        self.btn_spine.clicked.connect(self.on_export_spine)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("New skin prompt, e.g. black and gold dress")
        self.btn_skin = QPushButton("Generate Skin")
        self.btn_skin.clicked.connect(self.on_generate_skin)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        for w in (self.info, self.btn, self.btn_spine,
                  self.prompt_edit, self.btn_skin, self.log):
            layout.addWidget(w)
        root.setLayout(layout)
        self.setWidget(root)

    # required by DockWidget
    def canvasChanged(self, canvas):
        pass

    def say(self, msg):
        self.log.appendPlainText(str(msg))

    def _set_busy(self, busy):
        self.btn.setEnabled(not busy)
        self.btn_spine.setEnabled(not busy)
        self.btn_skin.setEnabled(not busy)

    # ------------------------------ split ------------------------------
    def on_split(self):
        self.log.clear()
        doc = Krita.instance().activeDocument()
        if doc is None:
            self.say("No document open. Open a flat character image first.")
            return
        if not os.path.exists(WORKFLOW_API):
            self.say("Workflow not found:\n  %s" % WORKFLOW_API)
            return
        try:
            comfy_client.ping(COMFY_URL)
        except Exception:
            self.say("Can't reach ComfyUI at %s - is it running on port 8188?" % COMFY_URL)
            return

        tmp = os.path.join(tempfile.gettempdir(), "autosplit_input.png")
        self.say("exporting active document (%dx%d)..." % (doc.width(), doc.height()))
        doc.setBatchmode(True)
        if not doc.exportImage(tmp, InfoObject()) or not os.path.exists(tmp):
            self.say("Failed to export the active document to PNG.")
            return

        # run the split on a worker thread; Krita stays responsive
        self._set_busy(True)
        self._worker = SplitWorker(COMFY_URL, WORKFLOW_API, tmp)
        self._worker.progress.connect(self.say)
        self._worker.done.connect(self._on_split_done)
        self._worker.failed.connect(self._on_split_failed)
        self._worker.start()

    def _on_split_failed(self, msg):
        self.say("ERROR: " + msg)
        self._set_busy(False)

    def _on_split_done(self):
        try:
            doc = Krita.instance().activeDocument()
            entries = comfy_client.read_parts(COMFY_OUTPUT_DIR, PART_SUBDIRS)
            if not entries:
                self.say("Split finished but no part metadata was found.")
                return
            transform = comfy_client.read_head_transform(COMFY_OUTPUT_DIR)
            if transform.get("scale", 1.0) != 1.0:
                self.say("facial transform: offset (%d,%d), scale %.2f"
                         % (transform["x_min"], transform["y_min"], transform["scale"]))
            self.say("importing %d parts as layers..." % len(entries))
            created = self._import_parts(doc, entries, transform)
            doc.refreshProjection()
            self.say("done - created %d layers." % created)
            self.info.setText("Created %d part layers." % created)
        except Exception as e:
            self.say("ERROR importing: %s" % e)
            self.say(traceback.format_exc())
        finally:
            self._set_busy(False)

    # --------------------------- layer import ---------------------------
    def _import_parts(self, doc, entries, transform):
        def zkey(p):
            try:
                return float(p.get("z_order", p.get("z_rank", 0)))
            except Exception:
                return 0.0

        body = sorted([e for e in entries if not e.get("_facial")], key=zkey, reverse=True)
        facial = sorted([e for e in entries if e.get("_facial")], key=zkey, reverse=True)

        sc = transform.get("scale", 1.0) or 1.0
        ox = transform.get("x_min", 0)
        oy = transform.get("y_min", 0)

        root = doc.rootNode()
        main_group = doc.createGroupLayer("character_parts")
        root.addChildNode(main_group, None)

        created = 0
        for p in body:
            if self._add_part(doc, main_group, p, sc, ox, oy):
                created += 1

        if facial:
            face_group = doc.createGroupLayer("face")
            main_group.addChildNode(face_group, None)  # on top of the body parts
            for p in facial:
                if self._add_part(doc, face_group, p, sc, ox, oy):
                    created += 1
        return created

    def _add_part(self, doc, parent, p, sc, ox, oy):
        try:
            path = p.get("_path")
            tag = str(p.get("tag", "part")).replace("/", "_")
            xyxy = p.get("xyxy") or [0, 0, 0, 0]

            img = QImage(path)
            if img.isNull():
                self.say("  skip '%s' (could not load %s)" % (tag, os.path.basename(path)))
                return False
            img = img.convertToFormat(QImage.Format_ARGB32)
            w, h = img.width(), img.height()

            if p.get("_facial"):
                if sc and sc != 1.0:
                    nw = max(1, int(round(w / sc)))
                    nh = max(1, int(round(h / sc)))
                    img = img.scaled(nw, nh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    img = img.convertToFormat(QImage.Format_ARGB32)
                    w, h = img.width(), img.height()
                x0 = int(ox + xyxy[0] / sc)
                y0 = int(oy + xyxy[1] / sc)
            else:
                x0, y0 = int(xyxy[0]), int(xyxy[1])

            ptr = img.bits()
            try:
                ptr.setsize(img.sizeInBytes())
            except AttributeError:
                ptr.setsize(img.byteCount())
            buf = QByteArray(bytes(ptr))

            node = doc.createNode(tag, "paintlayer")
            node.setPixelData(buf, x0, y0, w, h)
            parent.addChildNode(node, None)
            return True
        except Exception as e:
            self.say("  skip '%s' (%s)" % (p.get("tag"), e))
            return False

    # --------------------------- spine export ---------------------------
    def on_export_spine(self):
        self.log.clear()
        self.btn_spine.setEnabled(False)
        try:
            doc = Krita.instance().activeDocument()
            fn = doc.fileName() if doc else ""
            character = os.path.splitext(os.path.basename(fn))[0] if fn else "character"
            entries = comfy_client.read_parts(COMFY_OUTPUT_DIR, PART_SUBDIRS)
            if not entries:
                self.say("No split parts found - run Split Character first.")
                return
            transform = comfy_client.read_head_transform(COMFY_OUTPUT_DIR)
            self.say("exporting %d parts to Spine as '%s'..." % (len(entries), character))
            json_path = spine_export.export_to_spine(
                entries, transform, SPINE_OUT_DIR, character, SPINE_VERSION, progress=self.say)
            self.say("done. In Spine: File -> Import Data -> %s" % os.path.basename(json_path))
            self.info.setText("Spine project written to:\n%s" % SPINE_OUT_DIR)
        except Exception as e:
            self.say("ERROR: %s" % e)
            self.say(traceback.format_exc())
        finally:
            self.btn_spine.setEnabled(True)

    # --------------------------- skin generation ---------------------------
    def on_generate_skin(self):
        self.log.clear()
        doc = Krita.instance().activeDocument()
        if doc is None:
            self.say("No document open. Open a character first.")
            return
        try:
            comfy_client.ping(COMFY_URL)
        except Exception:
            self.say("Can't reach ComfyUI at %s - is it running on port 8188?" % COMFY_URL)
            return
        tmp = os.path.join(tempfile.gettempdir(), "restyle_input.png")
        self.say("exporting character for restyle...")
        doc.setBatchmode(True)
        if not doc.exportImage(tmp, InfoObject()) or not os.path.exists(tmp):
            self.say("Failed to export the document to PNG.")
            return
        prompt = self.prompt_edit.text()
        self._set_busy(True)
        self._restyle_worker = RestyleWorker(COMFY_URL, tmp, prompt, COMFY_OUTPUT_DIR)
        self._restyle_worker.progress.connect(self.say)
        self._restyle_worker.done.connect(self._on_restyle_done)
        self._restyle_worker.failed.connect(self._on_restyle_failed)
        self._restyle_worker.start()

    def _on_restyle_failed(self, msg):
        self.say("ERROR: " + msg)
        self._set_busy(False)

    def _on_restyle_done(self, path):
        try:
            if not path or not os.path.exists(path):
                self.say("Restyle finished but the output image wasn't found.")
                return
            self.say("opening restyled skin: %s" % os.path.basename(path))
            app = Krita.instance()
            new_doc = app.openDocument(path)
            app.activeWindow().addView(new_doc)
            self.info.setText("New skin opened. Split it, then Export to Spine.")
        except Exception as e:
            self.say("ERROR opening restyle: %s" % e)
            self.say(traceback.format_exc())
        finally:
            self._set_busy(False)
